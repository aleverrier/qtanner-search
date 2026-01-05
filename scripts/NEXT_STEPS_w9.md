# Next steps for a deeper w=9 search

This note is written for the current `scripts/run_search_w9_smallgroups.py` workflow and the
`scripts/run_one_group_w9*.py` wrappers.

## 1) Run a deeper search on a single group

Use the deep wrapper (it forwards arguments with more aggressive defaults: larger `max-*`,
`qd-num=2000`, and GAP session enabled):

```bash
python scripts/run_one_group_w9_deep.py --only-group 4,2 --slice-d-min 9
```

Notes:

* For `n=144`, `qd-num=2000` is a sensible “certification” budget.
* If you set `--slice-d-min` too aggressively, you can miss good quantum codes (including ones
  from the paper). When in doubt, lower `--slice-d-min` by 1–2 and compensate by increasing
  `--max-pairs`.

## 2) Increase coverage if you only see very small k

If you only see `k=2,4` (or similarly tiny dimensions), you likely need to increase search
coverage and/or loosen slice filtering:

```bash
python scripts/run_one_group_w9_deep.py \
  --only-group 4,2 \
  --slice-d-min 9 \
  --max-A 200 --max-B 200 --max-pairs 20000 \
  --a1v-max 30 --b1v-max 30
```

## 3) Refine QDistRnd for a specific candidate directory

When you have a promising candidate directory under `data/tmp/search_w9_smallgroups/...` that
contains `Hx.mtx` and `Hz.mtx`, re-run QDistRnd with more trials:

```bash
python scripts/refine_qdistrnd_dir.py data/tmp/search_w9_smallgroups/o4_g2_p123_a12_b20 \
  --qd-num 2000 --qd-mindist 12 --timeout 180
```

* Use `--qd-mindist` to trigger fast early-exit on bad instances.
* The script prints `dX`, `dZ`, and `d = min(dX,dZ)`.

## 4) When you find a leader, do one “extra confidence” run

If a candidate is clearly leading for its `(n,k)` bucket, you can bump to 10k on just that
candidate (still far below the 50k used in the paper):

```bash
python scripts/refine_qdistrnd_dir.py data/tmp/search_w9_smallgroups/<cand_dir> \
  --qd-num 10000 --qd-mindist <best_d_so_far>
```

## 5) Record the 4 classical slice-code parameters

When comparing quantum candidates, it’s useful to log the four classical codes in each slice
(type A and B). If you’re not already printing them in your candidate summary, add them to the
candidate JSON/TSV record that your search script writes, so you can later filter/sort by:

* `(nA, kA, dA)` for the two classical codes in type A slice
* `(nB, kB, dB)` for the two classical codes in type B slice

This is often a better “cheap score” than only keeping `min(d_slice)`.
