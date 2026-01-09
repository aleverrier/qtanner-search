#!/usr/bin/env python3
"""
Recover (A,B) and local permutations from LRZ paper .mtx parity-check matrices
for the w=9 construction (local [6,3,3] on both sides), as implemented in build_QT.py.

Single-code:
  python scripts/lrz_recover_w9.py --hx path/to/HX_*.mtx --hz path/to/HZ_*.mtx

Batch:
  python scripts/lrz_recover_w9.py --root data/lrz_paper_mtx/633x633 \
                                   --out  data/lrz_paper_mtx/633x633/recovered_codes.md
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# ----------------------------
# Local [6,3,3] blocks (paper / build_QT.py)
# ----------------------------

def H_local_633() -> np.ndarray:
    I = np.eye(3, dtype=np.uint8)
    J = np.ones((3, 3), dtype=np.uint8)
    return np.concatenate([I, (J ^ I)], axis=1) % 2

def G_local_633() -> np.ndarray:
    I = np.eye(3, dtype=np.uint8)
    J = np.ones((3, 3), dtype=np.uint8)
    return np.concatenate([(J ^ I), I], axis=1) % 2

H633 = H_local_633()
G633 = G_local_633()

def permute_cols(M: np.ndarray, sigma_1based: Sequence[int]) -> np.ndarray:
    idx = [s - 1 for s in sigma_1based]  # build_QT uses 1-based sigmas
    return (M[:, idx] & 1).astype(np.uint8)


# ----------------------------
# MatrixMarket (.mtx) I/O (binary matrices)
# ----------------------------

def read_mtx_binary(path: Path) -> np.ndarray:
    """
    Read MatrixMarket 'coordinate' format with integer entries (assumed 1s),
    returning a dense uint8 matrix mod 2.
    """
    with path.open("r") as f:
        header = f.readline()
        if not header.startswith("%%MatrixMarket"):
            raise ValueError(f"{path}: not a MatrixMarket file (bad header)")
        # Skip comments
        line = f.readline()
        while line.startswith("%"):
            line = f.readline()
        m, n, nnz = map(int, line.strip().split())
        M = np.zeros((m, n), dtype=np.uint8)
        for _ in range(nnz):
            line = f.readline()
            if not line:
                break
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            i = int(parts[0]) - 1
            j = int(parts[1]) - 1
            M[i, j] ^= 1  # XOR in case of duplicates
    return M


# ----------------------------
# Group via GAP: export mul/inv tables in AsList order
# ----------------------------

@dataclass(frozen=True)
class TableGroup:
    order: int
    identity: int
    mul_tab: List[List[int]]  # 1-based entries, accessed with [a-1][b-1]
    inv_tab: List[int]        # 1-based entries, accessed with [a-1]
    elts: Optional[List[str]] = None
    gap_expr: Optional[str] = None

    def mul(self, a: int, b: int) -> int:
        return self.mul_tab[a - 1][b - 1]

    def inv(self, a: int) -> int:
        return self.inv_tab[a - 1]

    def fmt(self, a: int) -> str:
        if self.elts and 1 <= a <= len(self.elts):
            return self.elts[a - 1]
        return str(a)


def _run_gap(code: str) -> str:
    proc = subprocess.run(
        ["gap", "-q"],
        input=code,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"GAP failed:\n{proc.stderr}")
    return proc.stdout.strip()


def gap_export_table(gap_expr: str, gap_cmd: str = "gap") -> "TableGroup":
    """
    Export group multiplication/inverse tables from GAP WITHOUT relying on JsonString.

    Why:
      - Your GAP doesn't define JsonString (even if LoadPackage("json") prints true).
      - GAP prints values of statements ending in ';' (must use ';;' to suppress).
      - We therefore print a small marker protocol and parse it line-by-line.

    Returns:
      A TableGroup (whatever fields your script defines) populated from the exported tables.
    """
    import ast as _ast
    import subprocess
    import textwrap
    import re
    import inspect

    # Build GAP program (use ';;' to suppress printing intermediate values)
    gap_code = textwrap.dedent(f"""
        G := {gap_expr};; 
        L := AsList(G);;
        n := Length(L);;
        id := Position(L, Identity(G));;

        mul := List([1..n], i -> List([1..n], j -> Position(L, L[i]*L[j])));;
        inv := List([1..n], i -> Position(L, L[i]^-1));;
        elts := List(L, String);;

        Print("__ORDER__");;    Print("\\n");; Print(n);;  Print("\\n");;
        Print("__IDENTITY__");; Print("\\n");; Print(id);; Print("\\n");;
        Print("__MUL__");;      Print("\\n");; Print(mul);; Print("\\n");;
        Print("__INV__");;      Print("\\n");; Print(inv);; Print("\\n");;
        Print("__ELTS__");;     Print("\\n");; Print(elts);; Print("\\n");;
        Print("__END__");;      Print("\\n");;

        QUIT;
    """).strip() + "\n"

    proc = subprocess.run(
        [gap_cmd, "-q"],
        input=gap_code,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    out = proc.stdout
    err = proc.stderr

    if proc.returncode != 0:
        raise RuntimeError(f"GAP failed (rc={proc.returncode}).\nSTDERR:\n{err}\nSTDOUT:\n{out}")

    marker_re = re.compile(r"^__([A-Z0-9]+)__$")

    blocks = {}
    cur = None
    buf = []
    for raw in out.splitlines():
        s = raw.strip()
        m = marker_re.match(s)
        if m:
            tag = m.group(1)
            if tag == "END":
                if cur is not None:
                    blocks[cur] = "\n".join(buf).strip()
                cur = None
                buf = []
                break

            if cur is not None:
                blocks[cur] = "\n".join(buf).strip()
            cur = tag
            buf = []
            continue

        if cur is not None:
            buf.append(raw)

    if cur is not None and cur not in blocks:
        blocks[cur] = "\n".join(buf).strip()

    required = ["ORDER", "IDENTITY", "MUL", "INV", "ELTS"]
    missing = [t for t in required if t not in blocks or blocks[t].strip() == ""]
    if missing:
        raise RuntimeError(
            f"Missing blocks {missing} in GAP output.\nSTDOUT:\n{out}\nSTDERR:\n{err}"
        )

    order = int(blocks["ORDER"].split()[0])
    identity = int(blocks["IDENTITY"].split()[0])
    mul_tab = _ast.literal_eval(blocks["MUL"])
    inv_tab = _ast.literal_eval(blocks["INV"])
    elts = _ast.literal_eval(blocks["ELTS"])

    # Build TableGroup robustly (field names may vary across versions of the script)
    values = {
        "order": order, "n": order,
        "identity": identity, "id": identity,
        "mul_tab": mul_tab, "mul": mul_tab, "mul_table": mul_tab,
        "inv_tab": inv_tab, "inv": inv_tab, "inv_table": inv_tab,
        "elts": elts, "elements": elts, "names": elts,
        "gap_expr": gap_expr, "expr": gap_expr, "name": gap_expr,
    }

    try:
        sig = inspect.signature(TableGroup)
        kwargs = {}
        for pname, p in sig.parameters.items():
            if pname in values:
                kwargs[pname] = values[pname]
        need = [pname for pname, p in sig.parameters.items()
                if p.default is inspect._empty and pname not in kwargs]
        if need:
            raise TypeError(f"TableGroup needs {need}, but exporter only has {sorted(values.keys())}")
        return TableGroup(**kwargs)
    except Exception as e:
        raise RuntimeError(f"Failed to construct TableGroup from GAP tables: {e}") from e
def gap_number_small_groups(order: int) -> int:
    gap_code = f"""
Print(NumberSmallGroups({order}));
QUIT;
"""
    out = _run_gap(gap_code)
    return int(out.strip())


def gap_smallgroup_expr(order: int, idx: int) -> str:
    return f"SmallGroup({order},{idx})"


# ----------------------------
# Build matrices (re-implementation of build_QT.build_css_mixed)
# ----------------------------

def build_css_mixed_from_params(Gp: TableGroup,
                                A: Sequence[int], B: Sequence[int],
                                sigA0: Sequence[int], sigA1: Sequence[int],
                                sigB0: Sequence[int], sigB1: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
    assert len(A) == 6 and len(B) == 6
    nG = Gp.order
    n = nG * 36

    H_A0 = permute_cols(H633, sigA0)
    H_A1 = permute_cols(H633, sigA1)
    G_A0 = permute_cols(G633, sigA0)
    G_A1 = permute_cols(G633, sigA1)

    H_B0 = permute_cols(H633, sigB0)
    H_B1 = permute_cols(H633, sigB1)
    G_B0 = permute_cols(G633, sigB0)
    G_B1 = permute_cols(G633, sigB1)

    def col_index(g: int, apos: int, bpos: int) -> int:
        # ((g-1)*6 + apos)*6 + bpos  (b fastest)
        return ((g - 1) * 6 + apos) * 6 + bpos

    X_rows: List[np.ndarray] = []
    # X @ V00: H_A0 ⊗ G_B0 (no group action)
    for g in range(1, nG + 1):
        for ja in range(3):
            for jb in range(3):
                row = np.zeros(n, dtype=np.uint8)
                for apos, a in enumerate(A):
                    if H_A0[ja, apos] == 0:
                        continue
                    for bpos, b in enumerate(B):
                        if G_B0[jb, bpos] == 0:
                            continue
                        g2 = g
                        row[col_index(g2, apos, bpos)] ^= 1
                X_rows.append(row)

    # X @ V11: H_A1 ⊗ G_B1 on (a^{-1} g b^{-1})
    for g in range(1, nG + 1):
        for ja in range(3):
            for jb in range(3):
                row = np.zeros(n, dtype=np.uint8)
                for apos, a in enumerate(A):
                    if H_A1[ja, apos] == 0:
                        continue
                    for bpos, b in enumerate(B):
                        if G_B1[jb, bpos] == 0:
                            continue
                        g2 = Gp.mul(Gp.mul(Gp.inv(a), g), Gp.inv(b))
                        row[col_index(g2, apos, bpos)] ^= 1
                X_rows.append(row)

    Z_rows: List[np.ndarray] = []
    # Z @ V01: G_A1 ⊗ H_B0 on (a^{-1} g)
    for g in range(1, nG + 1):
        for ia in range(3):
            for ib in range(3):
                row = np.zeros(n, dtype=np.uint8)
                for apos, a in enumerate(A):
                    if G_A1[ia, apos] == 0:
                        continue
                    for bpos, b in enumerate(B):
                        if H_B0[ib, bpos] == 0:
                            continue
                        g2 = Gp.mul(Gp.inv(a), g)
                        row[col_index(g2, apos, bpos)] ^= 1
                Z_rows.append(row)

    # Z @ V10: G_A0 ⊗ H_B1 on (g b^{-1})
    for g in range(1, nG + 1):
        for ia in range(3):
            for ib in range(3):
                row = np.zeros(n, dtype=np.uint8)
                for apos, a in enumerate(A):
                    if G_A0[ia, apos] == 0:
                        continue
                    for bpos, b in enumerate(B):
                        if H_B1[ib, bpos] == 0:
                            continue
                        g2 = Gp.mul(g, Gp.inv(b))
                        row[col_index(g2, apos, bpos)] ^= 1
                Z_rows.append(row)

    X = np.vstack(X_rows) % 2
    Z = np.vstack(Z_rows) % 2
    return X, Z


# ----------------------------
# Recovery helpers
# ----------------------------

def _col_tuple_w9(idx0: int) -> Tuple[int, int, int]:
    """0-based column -> (g, apos, bpos), with build_QT ordering."""
    g = idx0 // 36 + 1
    rem = idx0 % 36
    apos = rem // 6
    bpos = rem % 6
    return g, apos, bpos


def _project_support(row: np.ndarray, axis: str) -> Tuple[int, ...]:
    cols = np.flatnonzero(row)
    if axis == "apos":
        s = sorted({_col_tuple_w9(int(c))[1] for c in cols})
    elif axis == "bpos":
        s = sorted({_col_tuple_w9(int(c))[2] for c in cols})
    elif axis == "g":
        s = sorted({_col_tuple_w9(int(c))[0] for c in cols})
    else:
        raise ValueError(axis)
    return tuple(s)


def _find_sigma_ordered(obs_row_supports: Sequence[Tuple[int, ...]], base_matrix: np.ndarray) -> Optional[List[int]]:
    """
    Find 1-based sigma such that permute_cols(base_matrix, sigma) has row supports
    exactly obs_row_supports in row order.
    """
    import itertools
    for perm in itertools.permutations(range(1, 7)):
        M = permute_cols(base_matrix, perm)
        supps = [tuple(np.flatnonzero(M[r]).tolist()) for r in range(M.shape[0])]
        if supps == list(obs_row_supports):
            return list(perm)
    return None


@dataclass
class RecoveredW9:
    group: TableGroup
    A: List[int]
    B: List[int]
    sigA0: List[int]
    sigA1: List[int]
    sigB0: List[int]
    sigB1: List[int]
    verified: bool


def recover_w9_from_matrices(HX: np.ndarray, HZ: np.ndarray, Gp: TableGroup) -> Optional[RecoveredW9]:
    # Basic shape checks for w=9 (nA=nB=6)
    m, n = HX.shape
    if n % 36 != 0:
        return None
    nG = n // 36
    if m != 18 * nG or HZ.shape != (18 * nG, n):
        return None
    if Gp.order != nG:
        return None

    # --- Recover the 4 local sigmas (ordered) ---
    # X@V00 is the first 9*nG rows:
    start_x00 = 0
    obs_HA0 = [tuple(_project_support(HX[start_x00 + 3 * ja], "apos")) for ja in range(3)]
    obs_GB0 = [tuple(_project_support(HX[start_x00 + jb], "bpos")) for jb in range(3)]

    sigA0 = _find_sigma_ordered(obs_HA0, H633)
    sigB0 = _find_sigma_ordered(obs_GB0, G633)
    if sigA0 is None or sigB0 is None:
        return None

    # Z@V01 is the first 9*nG rows of HZ:
    start_z01 = 0
    obs_GA1 = [tuple(_project_support(HZ[start_z01 + 3 * ia], "apos")) for ia in range(3)]
    sigA1 = _find_sigma_ordered(obs_GA1, G633)
    if sigA1 is None:
        return None

    # Z@V10 is the second 9*nG rows of HZ:
    start_z10 = 9 * nG
    obs_HB1 = [tuple(_project_support(HZ[start_z10 + ib], "bpos")) for ib in range(3)]
    sigB1 = _find_sigma_ordered(obs_HB1, H633)
    if sigB1 is None:
        return None

    # --- Recover A and B via the group actions in Z blocks ---
    # Z@V01: g2 = inv(a) * g  => a = inv(g2 * inv(g))
    A_votes: List[Dict[int, int]] = [dict() for _ in range(6)]
    for r in range(0, 9 * nG):
        g = r // 9 + 1
        cols = np.flatnonzero(HZ[r])
        by_apos: Dict[int, set] = {}
        for c in cols:
            g2, apos, _bpos = _col_tuple_w9(int(c))
            by_apos.setdefault(apos, set()).add(g2)
        for apos, g2s in by_apos.items():
            if len(g2s) != 1:
                continue
            g2 = next(iter(g2s))
            inv_a = Gp.mul(g2, Gp.inv(g))
            a = Gp.inv(inv_a)
            A_votes[apos][a] = A_votes[apos].get(a, 0) + 1

    A: List[int] = []
    for apos in range(6):
        if not A_votes[apos]:
            return None
        a = max(A_votes[apos].items(), key=lambda kv: kv[1])[0]
        A.append(a)

    # Z@V10: g2 = g * inv(b)  => b = inv(inv(g) * g2)
    B_votes: List[Dict[int, int]] = [dict() for _ in range(6)]
    for r in range(9 * nG, 18 * nG):
        g = (r - 9 * nG) // 9 + 1
        cols = np.flatnonzero(HZ[r])
        by_bpos: Dict[int, set] = {}
        for c in cols:
            g2, _apos, bpos = _col_tuple_w9(int(c))
            by_bpos.setdefault(bpos, set()).add(g2)
        for bpos, g2s in by_bpos.items():
            if len(g2s) != 1:
                continue
            g2 = next(iter(g2s))
            inv_b = Gp.mul(Gp.inv(g), g2)
            b = Gp.inv(inv_b)
            B_votes[bpos][b] = B_votes[bpos].get(b, 0) + 1

    B: List[int] = []
    for bpos in range(6):
        if not B_votes[bpos]:
            return None
        b = max(B_votes[bpos].items(), key=lambda kv: kv[1])[0]
        B.append(b)

    # --- Recover sigB0 (optional cross-check) from X@V00; we already have sigB0 ---
    # Verified rebuild
    HX2, HZ2 = build_css_mixed_from_params(Gp, A, B, sigA0, sigA1, sigB0, sigB1)
    verified = bool(np.array_equal(HX, HX2) and np.array_equal(HZ, HZ2))

    return RecoveredW9(
        group=Gp, A=A, B=B,
        sigA0=sigA0, sigA1=sigA1, sigB0=sigB0, sigB1=sigB1,
        verified=verified,
    )


# ----------------------------
# Filename parsing + batch driver
# ----------------------------

_GROUP_TAG_RE = re.compile(r"^H[ XZ]_(?P<tag>.+)_(?P<n>\d+)_(?P<k>\d+)_(?P<d>\d+)\.mtx$", re.IGNORECASE)

def parse_group_tag_from_filename(path: Path) -> Optional[str]:
    m = _GROUP_TAG_RE.match(path.name)
    if not m:
        return None
    return m.group("tag")

def gap_expr_from_tag(tag: str) -> Optional[str]:
    m = re.fullmatch(r"C(\d+)", tag)
    if m:
        n = int(m.group(1))
        return f"CyclicGroup({n})"
    m = re.fullmatch(r"C(\d+)C(\d+)", tag)
    if m:
        a = int(m.group(1)); b = int(m.group(2))
        return f"DirectProduct(CyclicGroup({a}), CyclicGroup({b}))"
    if tag.upper() == "Q8":
        return "QuaternionGroup()"
    if tag.upper() == "C2C2":
        return "DirectProduct(CyclicGroup(2), CyclicGroup(2))"
    return None


def recover_one_pair(hx_path: Path, hz_path: Path, gap_expr: Optional[str] = None) -> Optional[RecoveredW9]:
    HX = read_mtx_binary(hx_path)
    HZ = read_mtx_binary(hz_path)

    if HX.shape != HZ.shape:
        raise ValueError(f"Shape mismatch: {hx_path} {HX.shape} vs {hz_path} {HZ.shape}")

    if HX.shape[1] % 36 != 0:
        return None
    nG = HX.shape[1] // 36

    candidates: List[str] = []
    if gap_expr:
        candidates = [gap_expr]
    else:
        tag = parse_group_tag_from_filename(hx_path) or ""
        guess = gap_expr_from_tag(tag)
        if guess:
            candidates.append(guess)
        else:
            # Fallback: try all SmallGroups of this order (works for small orders like 4,8,16,12,...)
            count = gap_number_small_groups(nG)
            candidates.extend([gap_smallgroup_expr(nG, i) for i in range(1, count + 1)])

    for expr in candidates:
        try:
            Gp = gap_export_table(expr)
        except Exception:
            continue
        rec = recover_w9_from_matrices(HX, HZ, Gp)
        if rec and rec.verified:
            return rec

    # Best-effort return (may be unverified if matrices were row-permuted externally)
    if candidates:
        Gp = gap_export_table(candidates[0])
        return recover_w9_from_matrices(HX, HZ, Gp)
    return None


def format_recovered_md(code_id: str, hx_path: Path, hz_path: Path, rec: RecoveredW9) -> str:
    Gp = rec.group
    A_idx = rec.A
    B_idx = rec.B
    A_str = [Gp.fmt(a) for a in A_idx]
    B_str = [Gp.fmt(b) for b in B_idx]
    return "\n".join([
        f"## {code_id}",
        "",
        f"- HX: `{hx_path}`",
        f"- HZ: `{hz_path}`",
        f"- |G|: **{Gp.order}** (identity index: {Gp.identity}; GAP expr: `{Gp.gap_expr}`)",
        f"- A (indices): `{A_idx}`",
        f"- A (GAP elts): `{A_str}`",
        f"- B (indices): `{B_idx}`",
        f"- B (GAP elts): `{B_str}`",
        f"- sigA0 (H_A0/G_A0 cols): `{rec.sigA0}`",
        f"- sigA1 (H_A1/G_A1 cols): `{rec.sigA1}`",
        f"- sigB0 (H_B0/G_B0 cols): `{rec.sigB0}`",
        f"- sigB1 (H_B1/G_B1 cols): `{rec.sigB1}`",
        f"- verified (rebuild == input): **{rec.verified}**",
        "",
    ])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hx", type=str, help="Path to HX_*.mtx (single-code mode)")
    ap.add_argument("--hz", type=str, help="Path to HZ_*.mtx (single-code mode)")
    ap.add_argument("--root", type=str, help="Directory to scan for HX_*.mtx / HZ_*.mtx pairs (batch mode)")
    ap.add_argument("--gap-expr", type=str, default=None, help="Explicit GAP expression for G (overrides filename inference)")
    ap.add_argument("--out", type=str, default=None, help="Output markdown file (batch mode); default: <root>/recovered_codes.md")
    ap.add_argument("--jsonl", type=str, default=None, help="Also write JSONL records here")
    args = ap.parse_args()

    if (args.hx and not args.hz) or (args.hz and not args.hx):
        raise SystemExit("Provide both --hx and --hz, or neither (and use --root).")

    if args.hx and args.hz:
        hx_path = Path(args.hx)
        hz_path = Path(args.hz)
        code_id = hx_path.stem.replace("HX_", "")
        rec = recover_one_pair(hx_path, hz_path, gap_expr=args.gap_expr)
        if rec is None:
            raise SystemExit("Not a w=9 matrix pair (n must be 36|G| and rows must be 18|G|), or recovery failed.")
        print(format_recovered_md(code_id, hx_path, hz_path, rec))
        return

    if not args.root:
        raise SystemExit("Provide either (--hx,--hz) or --root.")

    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"--root does not exist: {root}")

    out_path = Path(args.out) if args.out else (root / "recovered_codes.md")
    jsonl_path = Path(args.jsonl) if args.jsonl else None

    hx_files = list(root.rglob("HX_*.mtx"))
    pairs: List[Tuple[Path, Path]] = []
    for hx in hx_files:
        hz = hx.with_name(hx.name.replace("HX_", "HZ_", 1))
        if hz.exists():
            pairs.append((hx, hz))

    if not pairs:
        raise SystemExit(f"No HX_*.mtx / HZ_*.mtx pairs found under {root}")

    md_parts: List[str] = []
    md_parts.append("# Recovered LRZ w=9 instances (A=[6,3,3], B=[6,3,3])")
    md_parts.append("")
    md_parts.append(f"Scanned under `{root}`.")
    md_parts.append("")

    jsonl_f = None
    if jsonl_path:
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        jsonl_f = jsonl_path.open("w")

    ok = 0
    for hx, hz in sorted(pairs):
        code_id = hx.stem.replace("HX_", "")
        rec = recover_one_pair(hx, hz, gap_expr=args.gap_expr)

        if rec is None:
            # Skip non-w9 (e.g., your w=6/w=8 folders) silently
            continue

        md_parts.append(format_recovered_md(code_id, hx, hz, rec))
        if rec.verified:
            ok += 1

        if jsonl_f:
            Gp = rec.group
            record = {
                "code_id": code_id,
                "hx": str(hx),
                "hz": str(hz),
                "group": {
                    "order": Gp.order,
                    "identity": Gp.identity,
                    "gap_expr": Gp.gap_expr,
                },
                "A": rec.A,
                "B": rec.B,
                "A_elts": [Gp.fmt(a) for a in rec.A],
                "B_elts": [Gp.fmt(b) for b in rec.B],
                "sigA0": rec.sigA0,
                "sigA1": rec.sigA1,
                "sigB0": rec.sigB0,
                "sigB1": rec.sigB1,
                "verified": rec.verified,
            }
            jsonl_f.write(json.dumps(record) + "\n")

    if jsonl_f:
        jsonl_f.close()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(md_parts))
    print(f"Wrote: {out_path}  (verified {ok} codes)")


if __name__ == "__main__":
    main()
