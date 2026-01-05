"""qtanner: small utilities for local code handling."""

from .gf2 import gf2_is_in_rowspan, gf2_rank
from .group import FiniteGroup
from .lift_matrices import build_hx_hz, css_commutes
from .local_codes import (
    LocalCode,
    apply_col_perm_to_row,
    apply_col_perm_to_rows,
    distinct_column_permutation_representatives,
    hamming_6_3_3_shortened,
    hamming_8_4_4_extended,
    is_orthogonal,
    repetition_2,
    row_from_bits,
    span_codewords,
    variants_6_3_3,
    variants_8_4_4,
)
from .mtx import write_mtx, write_mtx_from_bitrows, write_mtx_from_rows
from .qdistrnd import dist_rand_css_mtx, gap_is_available, qdistrnd_is_available
from .qdistrnd import dist_rand_dz_mtx
from .slice_codes import (
    build_a_slice_checks_G,
    build_a_slice_checks_H,
    build_b_slice_checks_Gp,
    build_b_slice_checks_Hp,
)
from .trials import parse_trials_schedule, trials_for_n

__all__ = [
    "LocalCode",
    "row_from_bits",
    "apply_col_perm_to_row",
    "apply_col_perm_to_rows",
    "span_codewords",
    "is_orthogonal",
    "repetition_2",
    "hamming_6_3_3_shortened",
    "hamming_8_4_4_extended",
    "distinct_column_permutation_representatives",
    "variants_6_3_3",
    "variants_8_4_4",
    "gf2_rank",
    "gf2_is_in_rowspan",
    "FiniteGroup",
    "build_hx_hz",
    "css_commutes",
    "write_mtx",
    "write_mtx_from_rows",
    "write_mtx_from_bitrows",
    "dist_rand_css_mtx",
    "dist_rand_dz_mtx",
    "gap_is_available",
    "qdistrnd_is_available",
    "build_a_slice_checks_H",
    "build_a_slice_checks_G",
    "build_b_slice_checks_Hp",
    "build_b_slice_checks_Gp",
    "parse_trials_schedule",
    "trials_for_n",
]
