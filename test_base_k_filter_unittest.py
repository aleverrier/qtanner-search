import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qtanner.local_codes import hamming_6_3_3_shortened  # noqa: E402
from qtanner.progressive_search import (  # noqa: E402
    _build_variant_codes,
    _compute_base_k_table_for_variants,
    passes_base_k_filter,
)


class TestBaseKFilter(unittest.TestCase):
    def test_passes_base_k_filter(self) -> None:
        self.assertTrue(passes_base_k_filter(5, 0))
        self.assertTrue(passes_base_k_filter(6, 6))
        self.assertFalse(passes_base_k_filter(5, 6))

    def test_base_k_filter_subset(self) -> None:
        base_code = hamming_6_3_3_shortened()
        variants = _build_variant_codes(base_code)[:3]
        table = _compute_base_k_table_for_variants(
            base_code=base_code,
            variant_codes=variants,
        )
        max_k = max(max(row) for row in table)
        total = len(variants) * len(variants)
        kept_all = sum(
            1 for row in table for k_base in row if passes_base_k_filter(k_base, 0)
        )
        kept_none = sum(
            1
            for row in table
            for k_base in row
            if passes_base_k_filter(k_base, max_k + 1)
        )
        self.assertEqual(kept_all, total)
        self.assertEqual(kept_none, 0)


if __name__ == "__main__":
    unittest.main()
