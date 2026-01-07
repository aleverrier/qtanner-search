Quantum Tanner code search (Leverrier-Rozendaal-Zemor, arXiv:2512.20532).

Local code support is limited to the shortened Hamming [6,3,3].

Quick commands:
- Tiny end-to-end sanity check (asserts HX*HZ^T = 0):
  `python scripts/run_tiny_instance.py`
- Main search driver (SmallGroup search):
  `python scripts/run_search_w9_smallgroups.py --order-max 9 --n-min 100 --n-max 200`
- Run tests:
  `make test`
