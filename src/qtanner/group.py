"""Finite group helpers with 0-based index conventions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class FiniteGroup:
    """Finite group defined by a 0-based multiplication table.

    Index conventions:
    - Elements are identified by integers 0..order-1.
    - mul_table[a][b] gives the index of a * b in the group.
    - inv[x] gives the index of x^{-1}.
    """

    name: str
    order: int
    mul_table: List[List[int]]
    inv: List[int]

    def mul(self, a: int, b: int) -> int:
        """Multiply group elements by 0-based indices."""
        return self.mul_table[a][b]

    def inv_of(self, a: int) -> int:
        """Return the 0-based index of a^{-1}."""
        return self.inv[a]

    @classmethod
    def cyclic(cls, n: int) -> "FiniteGroup":
        """Cyclic group of order n using additive notation mod n."""
        mul_table = [[(i + j) % n for j in range(n)] for i in range(n)]
        inv = [(-i) % n for i in range(n)]
        return cls(name=f"C{n}", order=n, mul_table=mul_table, inv=inv)


__all__ = ["FiniteGroup"]
