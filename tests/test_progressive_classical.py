import random

from qtanner.group import CyclicGroup
from qtanner.local_codes import hamming_6_3_3_shortened
import qtanner.progressive_search as progressive_search


def test_abelian_precompute_skips_b_dist_m4ri(monkeypatch, tmp_path) -> None:
    group = CyclicGroup(2)
    base_code = hamming_6_3_3_shortened()
    variant_codes = [base_code]
    multisets = [
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1],
    ]
    rng = random.Random(0)
    calls = {"count": 0}

    def fake_classical_eval(rows, n_cols, *, steps, wmin, rng, dist_m4ri_cmd):
        calls["count"] += 1
        return {
            "k": 1,
            "rank": 0,
            "d_signed": 1,
            "d_est": 1,
            "early_stop": False,
            "seed": 0,
        }

    monkeypatch.setattr(progressive_search, "_classical_eval", fake_classical_eval)

    a_lookup = {}
    a_keep, a_hist, a_total = progressive_search._precompute_classical_side(
        side="A",
        group=group,
        multisets=multisets,
        variant_codes=variant_codes,
        base_code=base_code,
        steps=10,
        classical_target=1,
        dist_m4ri_cmd="dist_m4ri",
        rng=rng,
        out_path=tmp_path / "classical_A.jsonl",
        lookup=a_lookup,
        progress_every=1000,
        progress_seconds=1000.0,
    )
    b_keep, b_hist, b_total = progressive_search._precompute_classical_side_from_lookup(
        side="B",
        group=group,
        multisets=multisets,
        variant_codes=variant_codes,
        lookup=a_lookup,
        out_path=tmp_path / "classical_B.jsonl",
        progress_every=1000,
        progress_seconds=1000.0,
    )

    assert a_total == b_total
    assert calls["count"] == a_total * 2
    assert len(a_keep) == len(b_keep)
    assert a_hist == b_hist


def test_inverse_multiset_mapping_cyclic3() -> None:
    group = CyclicGroup(3)
    multiset = [0, 1, 1, 2, 2, 2]
    inv_multiset = progressive_search._inverse_multiset_key(group, multiset)
    assert inv_multiset == (0, 1, 1, 1, 2, 2)
