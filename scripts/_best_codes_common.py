from __future__ import annotations
from _ensure_python import ensure_minimum_python
ensure_minimum_python()

import json, os, re, shutil, tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00","Z")

def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))

def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=str(path.parent), encoding="utf-8") as f:
        f.write(text)
        tmp = f.name
    os.replace(tmp, path)

def atomic_write_json(path: Path, obj: Any, *, indent: int = 2, sort_keys: bool = False) -> None:
    text = json.dumps(obj, indent=indent, ensure_ascii=False, sort_keys=sort_keys)
    atomic_write_text(path, text + "\n")

def _to_int(v: Any) -> Optional[int]:
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, float) and v.is_integer():
        return int(v)
    if isinstance(v, str):
        s=v.strip()
        if s and re.fullmatch(r"-?\d+", s):
            try: return int(s)
            except: return None
    return None

def get_int(d: Any, keys: Iterable[str]) -> Optional[int]:
    if not isinstance(d, dict):
        return None
    for k in keys:
        if k in d:
            iv=_to_int(d.get(k))
            if iv is not None:
                return iv
    return None

_D_RE = re.compile(r"(.*)_d(\d+)$")

def code_id_without_d(code_id: str) -> str:
    m=_D_RE.match(code_id)
    return m.group(1) if m else code_id

def code_id_d_from_suffix(code_id: str) -> Optional[int]:
    m=_D_RE.match(code_id)
    return int(m.group(2)) if m else None

def code_id_with_d(code_id_no_d: str, d: int) -> str:
    m=_D_RE.match(code_id_no_d)
    if m:
        code_id_no_d=m.group(1)
    return f"{code_id_no_d}_d{int(d)}"

def extract_code_id(meta: Dict[str, Any], fallback: Optional[str]=None) -> str:
    cid = meta.get("code_id") or meta.get("id") or meta.get("name")
    if isinstance(cid, str) and cid:
        return cid
    return fallback or ""

def extract_group_spec(meta: Dict[str, Any]) -> Optional[str]:
    g = meta.get("group")
    if isinstance(g, dict):
        spec=g.get("spec")
        if isinstance(spec,str) and spec:
            return spec
    if isinstance(g,str) and g:
        return g
    A=meta.get("A")
    if isinstance(A, dict):
        g2=A.get("group")
        if isinstance(g2, dict):
            spec=g2.get("spec")
            if isinstance(spec,str) and spec:
                return spec
    return None

def extract_n(meta: Dict[str, Any]) -> Optional[int]:
    return _to_int(meta.get("n"))

def extract_k(meta: Dict[str, Any]) -> Optional[int]:
    return _to_int(meta.get("k"))

def extract_A_elems(meta: Dict[str, Any]) -> Optional[List[int]]:
    A = meta.get("A")
    if isinstance(A, dict) and isinstance(A.get("elements"), list):
        out=[]
        for x in A["elements"]:
            ix=_to_int(x)
            if ix is None: return None
            out.append(ix)
        return out
    if isinstance(meta.get("A_elems"), list):
        out=[]
        for x in meta["A_elems"]:
            ix=_to_int(x)
            if ix is None: return None
            out.append(ix)
        return out
    return None

def extract_B_elems(meta: Dict[str, Any]) -> Optional[List[int]]:
    B = meta.get("B")
    if isinstance(B, dict) and isinstance(B.get("elements"), list):
        out=[]
        for x in B["elements"]:
            ix=_to_int(x)
            if ix is None: return None
            out.append(ix)
        return out
    if isinstance(meta.get("B_elems"), list):
        out=[]
        for x in meta["B_elems"]:
            ix=_to_int(x)
            if ix is None: return None
            out.append(ix)
        return out
    return None

def extract_distance_bounds(meta: Dict[str, Any]) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    dist = meta.get("distance")
    if not isinstance(dist, dict):
        dist = {}
    d_ub = get_int(dist, ["d_ub","d","distance_ub","distance"])
    dX = get_int(dist, ["dX_best","dX_ub","dx_best","dx_ub","dX"])
    dZ = get_int(dist, ["dZ_best","dZ_ub","dz_best","dz_ub","dZ"])
    fast = dist.get("fast")
    if isinstance(fast, dict):
        fdx=fast.get("dx")
        fdz=fast.get("dz")
        if isinstance(fdx, dict) and dX is None:
            dX = get_int(fdx, ["d_ub","d","signed","d_est"])
        if isinstance(fdz, dict) and dZ is None:
            dZ = get_int(fdz, ["d_ub","d","signed","d_est"])
    cid = extract_code_id(meta, "")
    if d_ub is None:
        d_ub = code_id_d_from_suffix(cid) if cid else None
    if d_ub is None and (dX is not None or dZ is not None):
        ds=[x for x in (dX,dZ) if x is not None]
        if ds: d_ub=min(ds)
    return dX, dZ, d_ub

def extract_trials(meta: Dict[str, Any]) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    dist = meta.get("distance")
    if not isinstance(dist, dict):
        dist = {}
    total = get_int(dist, ["steps_used_total","steps_total","steps","trials","m4ri_steps"])
    sx = get_int(dist, ["steps_used_x","steps_x"])
    sz = get_int(dist, ["steps_used_z","steps_z"])
    fast = dist.get("fast")
    if isinstance(fast, dict):
        fdx=fast.get("dx"); fdz=fast.get("dz")
        if isinstance(fdx, dict) and sx is None: sx=get_int(fdx, ["steps"])
        if isinstance(fdz, dict) and sz is None: sz=get_int(fdz, ["steps"])
    if total is None and (sx is not None or sz is not None):
        if sx is not None and sz is not None: total=sx+sz
        else: total=sx if sx is not None else sz
    if total is None:
        total = get_int(meta, ["m4ri_trials","trials","m4ri_steps","steps_used_total","steps"])
    return total, sx, sz

def unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    base = path
    m = re.match(r"^(.*)__dup(\d+)$", base.name)
    stem = base.name
    start = 1
    if m:
        stem = m.group(1)
        start = int(m.group(2)) + 1
    for i in range(start, start + 10000):
        cand = base.with_name(f"{stem}__dup{i}")
        if not cand.exists():
            return cand
    raise RuntimeError(f"Could not find unique path for {path}")

def safe_move(src: Path, dst: Path) -> Path:
    if not src.exists():
        return dst
    dst = unique_path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))
    return dst

def matrices_for_code(best_dir: Path, code_id: str) -> List[Path]:
    mats_dir = best_dir / "matrices"
    if not mats_dir.exists():
        return []
    return sorted([p for p in mats_dir.iterdir() if p.is_file() and p.name.startswith(code_id) and p.suffix.lower()==".mtx"])

def code_paths(best_dir: Path, code_id: str) -> Dict[str, Path]:
    return {
        "meta": best_dir / "meta" / f"{code_id}.json",
        "collected_dir": best_dir / "collected" / code_id,
        "collected_meta": best_dir / "collected" / code_id / "meta.json",
        "matrices_dir": best_dir / "matrices",
    }

def archive_code(best_dir: Path, code_id: str, archive_root: Path) -> None:
    p=code_paths(best_dir, code_id)
    safe_move(p["meta"], archive_root / "meta" / p["meta"].name)
    safe_move(p["collected_dir"], archive_root / "collected" / code_id)
    for m in matrices_for_code(best_dir, code_id):
        safe_move(m, archive_root / "matrices" / m.name)

def rename_code(best_dir: Path, old_id: str, new_id: str, *, archive_collisions_root: Optional[Path]=None) -> None:
    if old_id == new_id:
        return
    old=code_paths(best_dir, old_id)
    new=code_paths(best_dir, new_id)
    collision = new["meta"].exists() or new["collected_dir"].exists() or any(m.exists() for m in matrices_for_code(best_dir, new_id))
    if collision:
        if archive_collisions_root is None:
            raise RuntimeError(f"Rename collision {old_id}->{new_id} without archive root")
        archive_code(best_dir, new_id, archive_collisions_root / new_id)
    if old["meta"].exists():
        new["meta"].parent.mkdir(parents=True, exist_ok=True)
        os.replace(old["meta"], new["meta"])
    if old["collected_dir"].exists():
        new["collected_dir"].parent.mkdir(parents=True, exist_ok=True)
        os.replace(old["collected_dir"], new["collected_dir"])
    for m in matrices_for_code(best_dir, old_id):
        new_name = new_id + m.name[len(old_id):]
        os.replace(m, m.with_name(new_name))

@dataclass(frozen=True)
class CodeScore:
    d_ub: Optional[int]
    trials_total: Optional[int]
    def sort_key_conservative(self) -> Tuple[int,int]:
        d = self.d_ub if self.d_ub is not None else 10**18
        t = self.trials_total if self.trials_total is not None else -1
        return (d, -t)
