from qtanner.classical_dist_fast import _estimate_classical_distance_fast_details
from qtanner.classical_dist_fast import estimate_classical_distance_fast


def test_fast_distance_exact_small_code() -> None:
    # Parity-check for length-3 repetition code: codewords {000, 111}
    h_rows = [0b011, 0b110]
    witness, k, exact = estimate_classical_distance_fast(
        h_rows_bits=h_rows,
        n=3,
        wmin=0,
        exhaustive_k_max=8,
        sample_count=16,
        rng_seed=0,
    )
    assert witness == 3
    assert k == 1
    assert exact is True


def test_fast_distance_sampling_early_abort() -> None:
    h_rows = []
    witness, k, exact, checked = _estimate_classical_distance_fast_details(
        h_rows_bits=h_rows,
        n=9,
        wmin=9,
        exhaustive_k_max=8,
        sample_count=5,
        rng_seed=123,
    )
    assert k == 9
    assert exact is False
    assert witness is not None
    assert witness <= 9
    assert checked == 1
