import argparse

from qtanner.mtx import write_mtx_from_bitrows
from qtanner.search import _process_dist_m4ri_batch


def test_target_distance_filter_and_best_by_k(tmp_path) -> None:
    args = argparse.Namespace(
        steps=10,
        target_distance=16,
        dist_m4ri_cmd="dist_m4ri",
    )
    outdir = tmp_path / "run"
    tmp_root = outdir / "tmp"
    promising_root = outdir / "promising"
    best_codes_root = outdir / "best_codes"
    outdir.mkdir(parents=True, exist_ok=True)
    tmp_root.mkdir(parents=True, exist_ok=True)
    promising_root.mkdir(parents=True, exist_ok=True)
    best_codes_root.mkdir(parents=True, exist_ok=True)
    results_path = outdir / "candidates.jsonl"

    coverage_by_group = {
        "C2xC2xC2": {"Q_meas_run": 0},
    }
    summary_records = []
    best_by_k = {}
    best_measured_by_k = {}
    best_survivor_by_k = {}

    def make_entry(candidate_id: str) -> dict:
        return {
            "candidate_id": candidate_id,
            "group": {"spec": "C2xC2xC2"},
            "n": 10,
            "k": 1,
            "local_codes": {"permA_idx": 0, "permB_idx": 0},
            "classical_slices": {},
            "base": {"n_base": 36, "k_base": 0},
            "seed": 1,
        }

    hx_rows = [0b1]
    hz_rows = [0b1]
    n_cols = 1

    batch_items = []
    for idx, candidate_id in enumerate(["cand1", "cand2", "cand3"]):
        cand_dir = tmp_root / candidate_id
        cand_dir.mkdir(parents=True, exist_ok=True)
        hx_path = cand_dir / "Hx.mtx"
        hz_path = cand_dir / "Hz.mtx"
        write_mtx_from_bitrows(str(hx_path), hx_rows, n_cols)
        write_mtx_from_bitrows(str(hz_path), hz_rows, n_cols)
        batch_items.append(
            (
                idx,
                hx_rows,
                hz_rows,
                n_cols,
                str(hx_path),
                str(hz_path),
                make_entry(candidate_id),
            )
        )

    distances = [15, 18, 17, 18, 17]
    calls = []

    def fake_estimator(hx, hz, n, steps, wmin, seed, dist_m4ri_cmd):
        calls.append(wmin)
        return distances.pop(0)

    with results_path.open("w", encoding="utf-8") as results_file:
        _process_dist_m4ri_batch(
            batch_items=batch_items,
            args=args,
            seed=1,
            outdir=outdir,
            tmp_root=tmp_root,
            promising_root=promising_root,
            best_codes_root=best_codes_root,
            results_file=results_file,
            summary_records=summary_records,
            best_by_k=best_by_k,
            best_measured_by_k=best_measured_by_k,
            best_survivor_by_k=best_survivor_by_k,
            coverage_by_group=coverage_by_group,
            estimator=fake_estimator,
        )

    assert calls[0] == 15
    assert len(calls) == 5
    assert len(best_by_k) == 1
    best_entry = next(iter(best_by_k.values()))
    assert best_entry["candidate_id"] == "cand2"

    lines = results_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    best_dir = best_codes_root / "C2xC2xC2" / "n10" / "k1" / "cand2"
    assert best_dir.exists()
