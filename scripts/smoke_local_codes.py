"""Quick smoke checks for local code variants."""
from __future__ import annotations
from _ensure_python import ensure_minimum_python
ensure_minimum_python()



import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from qtanner.local_codes import (
    hamming_6_3_3_shortened,
    is_orthogonal,
    variants_6_3_3,
)


def main() -> int:
    code_6 = hamming_6_3_3_shortened()
    print(f"[6,3,3] orthogonal: {is_orthogonal(code_6.H_rows, code_6.G_rows)}")

    v6 = variants_6_3_3()
    print(f"[6,3,3] variants: {len(v6)}")

    if len(v6) != 30:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
