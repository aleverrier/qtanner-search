#!/usr/bin/env python3
from __future__ import annotations
from _ensure_python import ensure_minimum_python
ensure_minimum_python()


import argparse
import itertools
import re
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np

# Reuse your working MatrixMarket reader + GAP exporter from lrz_recover_w9.py
# Load helpers from scripts/lrz_recover_w9.py WITHOUT requiring 'scripts' to be importable.
# This makes the script work when executed as: ./scripts/py scripts/lrz_recover_b2.py ...
import importlib.util as _ilu

_w9_path = Path(__file__).resolve().with_name("lrz_recover_w9.py")
_spec = _ilu.spec_from_file_location("lrz_recover_w9", _w9_path)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Could not load module spec for {_w9_path}")
_w9 = _ilu.module_from_spec(_spec)
import sys as _sys
_sys.modules[_spec.name] = _w9
_spec.loader.exec_module(_w9)  # type: ignore[attr-defined]

read_mtx_binary = _w9.read_mtx_binary
gap_export_table = _w9.gap_export_table



# ---------------------------------------------------------------------------
# Canonical local matrices from the LRZ paper (Appendix A.2)
# ---------------------------------------------------------------------------

# H[6,3,3], G[6,3,3]
H633 = np.array(
    [
        [1, 0, 0, 0, 1, 1],
        [0, 1, 0, 1, 0, 1],
        [0, 0, 1, 1, 1, 0],
    ],
    dtype=np.uint8,
)
G633 = np.array(
    [
        [0, 1, 1, 1, 0, 0],
        [1, 0, 1, 0, 1, 0],
        [1, 1, 0, 0, 0, 1],
    ],
    dtype=np.uint8,
)

# H[8,4,4], G[8,4,4]
H844 = np.array(
    [
        [1, 0, 0, 0, 0, 1, 1, 1],
        [0, 1, 0, 0, 1, 0, 1, 1],
        [0, 0, 1, 0, 1, 1, 0, 1],
        [0, 0, 0, 1, 1, 1, 1, 0],
    ],
    dtype=np.uint8,
)
G844 = np.array(
    [
        [0, 1, 1, 1, 1, 0, 0, 0],
        [1, 0, 1, 1, 0, 1, 0, 0],
        [1, 1, 0, 1, 0, 0, 1, 0],
        [1, 1, 1, 0, 0, 0, 0, 1],
    ],
    dtype=np.uint8,
)

# [2,1,2] repetition: H = G = [1 1]
HREP = np.array([[1, 1]], dtype=np.uint8)
GREP = np.array([[1, 1]], dtype=np.uint8)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gp_fields(Gp: Any) -> Tuple[int, int, List[List[int]], List[int], List[str], str]:
    """
    Robust to either:
      - dict-style: {"mul":..., "inv":..., "identity":..., "elts":...}
      - dataclass-style: .mul_tab, .inv_tab, .identity, .elts
    """
    gap_expr = getattr(Gp, "gap_expr", None) if not isinstance(Gp, dict) else Gp.get("gap_expr")
    if gap_expr is None:
        gap_expr = "(unknown GAP expr)"

    if isinstance(Gp, dict):
        mul = Gp.get("mul") or Gp.get("mul_tab")
        inv = Gp.get("inv") or Gp.get("inv_tab")
        identity = int(Gp.get("identity", 1))
        elts = Gp.get("elts") or []
        order = int(Gp.get("order", len(mul) if mul is not None else 0))
    else:
        mul = getattr(Gp, "mul_tab", None) or getattr(Gp, "mul", None)
        inv = getattr(Gp, "inv_tab", None) or getattr(Gp, "inv", None)
        identity = int(getattr(Gp, "identity", 1))
        elts = getattr(Gp, "elts", None) or []
        order = int(getattr(Gp, "order", len(mul) if mul is not None else 0))

    if mul is None or inv is None:
        raise RuntimeError("gap_export_table returned a group object without mul/inv tables")

    if order <= 0:
        order = len(mul)

    return order, identity, mul, inv, elts, str(gap_expr)


def gp_mul(mul_tab: List[List[int]], x: int, y: int) -> int:
    return mul_tab[x - 1][y - 1]  # 1-based


def gp_inv(inv_tab: List[int], x: int) -> int:
    return inv_tab[x - 1]  # 1-based


def permute_cols(M: np.ndarray, sigma_1based: List[int]) -> np.ndarray:
    idx = [s - 1 for s in sigma_1based]
    return M[:, idx]


def col_decode(c0: int, nA: int) -> Tuple[int, int, int]:
    """
    Inverse of build_QT.py col_index when nB=2:
        col = ((g-1)*nA + apos)*2 + bpos
    returns (g in 1..|G|, apos in 0..nA-1, bpos in {0,1})
    """
    bpos = c0 & 1
    t = c0 >> 1
    apos = t % nA
    g = t // nA + 1
    return g, apos, bpos


def infer_nA(HX: np.ndarray, user_nA: Optional[int]) -> int:
    """
    For LRZ instances with B=[2,1,2], every HX row has weight nA (6 or 8),
    since wt(row(H_A*))=nA/2 and repetition contributes a factor 2.
    """
    if user_nA is not None:
        return int(user_nA)
    w = int(np.median(HX.sum(axis=1)))
    if w in (6, 8):
        return w
    raise RuntimeError(
        f"Could not infer nA from HX row weights (median={w}). Pass --nA 6 or --nA 8."
    )


def parse_code_id_from_filename(hx_path: Path, m: int) -> str:
    m0 = re.match(r"^HX_(\d+)_(\d+)_(\d+)\.mtx$", hx_path.name)
    if not m0:
        return hx_path.stem.replace("HX_", f"C{m}_")
    n, k, d = m0.group(1), m0.group(2), m0.group(3)
    return f"C{m}_{n}_{k}_{d}"


def row_masks_ordered_from_block(mat: np.ndarray, rows_per_g: int, nA: int, g_choose: int) -> List[int]:
    """
    Extract local row patterns as bitmasks (ordered by local-row index),
    ignoring g2 and bpos: just which apos appear.
    """
    masks: List[int] = []
    base = (g_choose - 1) * rows_per_g
    for i in range(rows_per_g):
        cols = np.flatnonzero(mat[base + i])
        apos_set = {col_decode(int(c), nA)[1] for c in cols}
        mask = 0
        for apos in apos_set:
            mask |= 1 << apos
        masks.append(mask)
    return masks


def masks_ordered_from_matrix_rows(M: np.ndarray) -> List[int]:
    masks: List[int] = []
    for r in range(M.shape[0]):
        mask = 0
        for j in range(M.shape[1]):
            if int(M[r, j]) & 1:
                mask |= 1 << j
        masks.append(mask)
    return masks


def find_sigma_ordered(target_H_masks: List[int], target_G_masks: List[int], H_local: np.ndarray, G_local: np.ndarray) -> List[int]:
    nA = H_local.shape[1]
    for sigma in itertools.permutations(range(1, nA + 1)):
        sigma_l = list(sigma)
        Hperm = permute_cols(H_local, sigma_l)
        Gperm = permute_cols(G_local, sigma_l)
        if masks_ordered_from_matrix_rows(Hperm) != target_H_masks:
            continue
        if masks_ordered_from_matrix_rows(Gperm) != target_G_masks:
            continue
        return sigma_l
    raise RuntimeError("Could not find a column permutation sigma matching extracted local rows.")


def _find_g2_in_row(row_vec: np.ndarray, nA: int, apos: int, bpos: int) -> Optional[int]:
    cols = np.flatnonzero(row_vec)
    g2s = []
    for c in cols:
        g2, apos2, bpos2 = col_decode(int(c), nA)
        if apos2 == apos and bpos2 == bpos:
            g2s.append(g2)
    if len(g2s) != 1:
        return None
    return g2s[0]


def _extract_perm(block_mat: np.ndarray, rows_per_g: int, nA: int, ia: int, apos: int, bpos: int) -> List[int]:
    """
    Extract permutation p(g)=g2 from a single active (ia, apos, bpos) block.
    Assumes build_QT.py row order: g outer, then local-row index ia.
    """
    m = block_mat.shape[0] // rows_per_g
    p = [0] * (m + 1)  # 1-based; p[0] unused
    for g in range(1, m + 1):
        row_idx = (g - 1) * rows_per_g + ia
        g2 = _find_g2_in_row(block_mat[row_idx], nA, apos, bpos)
        if g2 is None:
            raise RuntimeError(f"Block (ia={ia}, apos={apos}, bpos={bpos}) not one-hot for g={g}.")
        p[g] = g2
    return p


def recover_b2_pair(hx_path: Path, hz_path: Path, *, gap_expr: Optional[str], gap_cmd: str, nA_user: Optional[int]) -> dict:
    HX = read_mtx_binary(hx_path)
    HZ = read_mtx_binary(hz_path)

    if HX.shape[1] != HZ.shape[1]:
        raise RuntimeError(f"HX/HZ column mismatch: HX {HX.shape}, HZ {HZ.shape}")

    nA = infer_nA(HX, nA_user)
    nB = 2
    n = HX.shape[1]

    if n % (nA * nB) != 0:
        raise RuntimeError(f"ncols={n} is not divisible by 2*nA={nA*nB}")
    m = n // (nA * nB)

    if gap_expr is None:
        gap_expr = f"CyclicGroup({m})"

    Gp = gap_export_table(gap_expr, gap_cmd=gap_cmd)
    order, identity, mul_tab, inv_tab, elts, gap_expr_used = _gp_fields(Gp)

    if order != m:
        raise RuntimeError(f"|G| mismatch: inferred {m} from ncols, but GAP expr yields {order}")

    if HX.shape[0] % (2 * m) != 0 or HZ.shape[0] % (2 * m) != 0:
        raise RuntimeError(f"Row counts not compatible with 2*|G|: HX {HX.shape}, HZ {HZ.shape}")

    rA = HX.shape[0] // (2 * m)
    kA = HZ.shape[0] // (2 * m)

    # Split blocks in build_QT.py order:
    #   HX = [X@V00 ; X@V11]
    #   HZ = [Z@V01 ; Z@V10]
    HX0 = HX[: rA * m, :]
    HX1 = HX[rA * m :, :]
    HZ_LA = HZ[: kA * m, :]
    HZ_RB = HZ[kA * m :, :]

    # -----------------------
    # Recover B from Z@V10: g2 = g * inv(b)
    # -----------------------
    B: List[int] = []
    for bpos in range(2):
        found: Optional[Tuple[int, int]] = None
        g_id = identity
        for ia in range(kA):
            row_idx = (g_id - 1) * kA + ia
            cols = np.flatnonzero(HZ_RB[row_idx])
            for c in cols:
                _, apos2, bpos2 = col_decode(int(c), nA)
                if bpos2 == bpos:
                    found = (ia, apos2)
                    break
            if found is not None:
                break
        if found is None:
            raise RuntimeError(f"Could not find any active (ia,apos) for bpos={bpos} in Z@V10 block")

        ia0, apos0 = found
        p = _extract_perm(HZ_RB, kA, nA, ia0, apos0, bpos)

        inv_b = p[identity]          # p(e) = inv(b)
        b = gp_inv(inv_tab, inv_b)   # b = inv(inv_b)

        for g in range(1, m + 1):
            if p[g] != gp_mul(mul_tab, g, inv_b):
                raise RuntimeError(f"RB mismatch for bpos={bpos} at g={g}")
        B.append(b)

    # -----------------------
    # Recover A from Z@V01: g2 = inv(a) * g
    # -----------------------
    A: List[int] = []
    for apos in range(nA):
        ia_found: Optional[int] = None
        row_base = (identity - 1) * kA
        for ia in range(kA):
            g2 = _find_g2_in_row(HZ_LA[row_base + ia], nA, apos, 0)
            if g2 is not None:
                ia_found = ia
                break
        if ia_found is None:
            raise RuntimeError(f"Could not find active ia for apos={apos} in Z@V01 block")

        p = _extract_perm(HZ_LA, kA, nA, ia_found, apos, 0)
        inv_a = p[identity]          # p(e) = inv(a)
        a = gp_inv(inv_tab, inv_a)   # a = inv(inv_a)

        for g in range(1, m + 1):
            if p[g] != gp_mul(mul_tab, inv_a, g):
                raise RuntimeError(f"LA mismatch for apos={apos} at g={g}")
        A.append(a)

    # -----------------------
    # Recover sigA0 and sigA1 relative to canonical matrices
    # -----------------------
    if nA == 6:
        H_local, G_local = H633, G633
    elif nA == 8:
        H_local, G_local = H844, G844
    else:
        raise RuntimeError(f"Unsupported nA={nA} (expected 6 or 8)")

    H0_masks = row_masks_ordered_from_block(HX0, rA, nA, identity)
    H1_masks = row_masks_ordered_from_block(HX1, rA, nA, identity)
    G1_masks = row_masks_ordered_from_block(HZ_LA, kA, nA, identity)
    G0_masks = row_masks_ordered_from_block(HZ_RB, kA, nA, identity)

    sigA0 = find_sigma_ordered(H0_masks, G0_masks, H_local, G_local)
    sigA1 = find_sigma_ordered(H1_masks, G1_masks, H_local, G_local)

    sigB0 = [1, 2]
    sigB1 = [1, 2]

    # -----------------------
    # Verify by rebuilding (same ordering as build_QT.py)
    # -----------------------
    def build_css_rep(A_idx: List[int], B_idx: List[int], sigA0: List[int], sigA1: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        H_A0 = permute_cols(H_local, sigA0)
        H_A1 = permute_cols(H_local, sigA1)
        G_A0 = permute_cols(G_local, sigA0)
        G_A1 = permute_cols(G_local, sigA1)

        ncols = m * nA * 2

        def col_index(g: int, apos0: int, bpos0: int) -> int:
            return ((g - 1) * nA + apos0) * 2 + bpos0

        X = np.zeros((2 * rA * m, ncols), dtype=np.uint8)
        Z = np.zeros((2 * kA * m, ncols), dtype=np.uint8)

        # X @ V00: g2 = g
        rr = 0
        for g in range(1, m + 1):
            for ia in range(rA):
                for ib in range(GREP.shape[0]):  # 1
                    for apos0, a in enumerate(A_idx):
                        if H_A0[ia, apos0] == 0:
                            continue
                        for bpos0, b in enumerate(B_idx):
                            if GREP[ib, bpos0] == 0:
                                continue
                            X[rr, col_index(g, apos0, bpos0)] ^= 1
                    rr += 1

        # X @ V11: g2 = inv(a) g inv(b)
        for g in range(1, m + 1):
            for ia in range(rA):
                for ib in range(GREP.shape[0]):
                    for apos0, a in enumerate(A_idx):
                        if H_A1[ia, apos0] == 0:
                            continue
                        for bpos0, b in enumerate(B_idx):
                            if GREP[ib, bpos0] == 0:
                                continue
                            g2 = gp_mul(mul_tab, gp_mul(mul_tab, gp_inv(inv_tab, a), g), gp_inv(inv_tab, b))
                            X[rr, col_index(g2, apos0, bpos0)] ^= 1
                    rr += 1

        # Z @ V01: g2 = inv(a) g
        rr = 0
        for g in range(1, m + 1):
            for ia in range(kA):
                for ib in range(HREP.shape[0]):  # 1
                    for apos0, a in enumerate(A_idx):
                        if G_A1[ia, apos0] == 0:
                            continue
                        for bpos0, b in enumerate(B_idx):
                            if HREP[ib, bpos0] == 0:
                                continue
                            g2 = gp_mul(mul_tab, gp_inv(inv_tab, a), g)
                            Z[rr, col_index(g2, apos0, bpos0)] ^= 1
                    rr += 1

        # Z @ V10: g2 = g inv(b)
        for g in range(1, m + 1):
            for ia in range(kA):
                for ib in range(HREP.shape[0]):
                    for apos0, a in enumerate(A_idx):
                        if G_A0[ia, apos0] == 0:
                            continue
                        for bpos0, b in enumerate(B_idx):
                            if HREP[ib, bpos0] == 0:
                                continue
                            g2 = gp_mul(mul_tab, g, gp_inv(inv_tab, b))
                            Z[rr, col_index(g2, apos0, bpos0)] ^= 1
                    rr += 1

        return X, Z

    HX2, HZ2 = build_css_rep(A, B, sigA0, sigA1)
    verified = bool(np.array_equal(HX2, HX) and np.array_equal(HZ2, HZ))

    code_id = parse_code_id_from_filename(hx_path, m)

    def el(idx: int) -> str:
        if 1 <= idx <= len(elts):
            return elts[idx - 1]
        return str(idx)

    return {
        "code_id": code_id,
        "hx_path": str(hx_path),
        "hz_path": str(hz_path),
        "nA": nA,
        "nB": 2,
        "rA": rA,
        "kA": kA,
        "group_order": m,
        "group_identity": identity,
        "gap_expr": gap_expr_used,
        "A_idx": A,
        "A_elts": [el(x) for x in A],
        "B_idx": B,
        "B_elts": [el(x) for x in B],
        "sigA0": sigA0,
        "sigA1": sigA1,
        "sigB0": sigB0,
        "sigB1": sigB1,
        "verified": verified,
    }


def print_markdown(rec: dict) -> None:
    print(f"## {rec['code_id']}\n")
    print(f"- HX: `{rec['hx_path']}`")
    print(f"- HZ: `{rec['hz_path']}`")
    print(f"- local codes: A=[{rec['nA']},{rec['nA']//2},{rec['nA']//2}]  B=[2,1,2] (repetition)")
    print(f"- |G|: **{rec['group_order']}** (identity index: {rec['group_identity']}; GAP expr: `{rec['gap_expr']}`)")
    print(f"- A (indices): `{rec['A_idx']}`")
    print(f"- A (GAP elts): `{rec['A_elts']}`")
    print(f"- B (indices): `{rec['B_idx']}`")
    print(f"- B (GAP elts): `{rec['B_elts']}`")
    print(f"- sigA0 (columns for H_A0/G_A0): `{rec['sigA0']}`")
    print(f"- sigA1 (columns for H_A1/G_A1): `{rec['sigA1']}`")
    print(f"- sigB0/sigB1: `{rec['sigB0']}` / `{rec['sigB1']}` (repetition: permutations don't matter)")
    print(f"- verified (rebuild == input): **{rec['verified']}**\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Recover A,B for QT/LRZ codes when B is the [2,1,2] repetition code.")
    ap.add_argument("--hx", type=Path, required=True)
    ap.add_argument("--hz", type=Path, required=True)
    ap.add_argument("--gap-expr", type=str, default=None, help="Optional GAP expr for the group. Default: CyclicGroup(|G|).")
    ap.add_argument("--gap-cmd", type=str, default="gap", help="GAP executable (default: gap)")
    ap.add_argument("--nA", type=int, default=None, help="Force nA (6 or 8). Normally inferred from HX row weight.")
    args = ap.parse_args()

    rec = recover_b2_pair(args.hx, args.hz, gap_expr=args.gap_expr, gap_cmd=args.gap_cmd, nA_user=args.nA)
    print_markdown(rec)


if __name__ == "__main__":
    main()
