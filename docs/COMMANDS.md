# qtanner-search: command cheat sheet

## 0) Environment
cd ~/qtanner-search
source .venv/bin/activate
pip install -e .
make test

## 1) Run a search (example: compare order-4 groups)
OUT="runs/C4_vs_C2xC2_$(date +%Y%m%d_%H%M%S)"

python -m qtanner.search \
  --groups C4,C2xC2 \
  --max-n 200 \
  --A-enum multiset \
  --B-enum multiset \
  --permH1 30 --permH1p 30 \
  --steps 2000 \
  --batch-size 200 \
  --seed 1 \
  --outdir "$OUT" \
  --gap-cmd gap \
  --dist-m4ri-cmd dist_m4ri

Notes:
- A/B multisets are deduplicated up to Aut(G) (Cayley-unique) before scoring.
- Enumeration modes: subset (default for large groups), multiset (with repetition), ordered (ordered tuples).
- Use `--target-distance` to reject candidates below a target distance (sets `wmin=target-1`).

## 1b) Progressive exhaustive classical-first search
OUT="results/progressive_c2xc2xc2_d16_$(date +%Y%m%d_%H%M%S)"

python -m qtanner.search progressive \
  --group C2xC2xC2 \
  --target-distance 16 \
  --classical-steps 100 \
  --quantum-steps-fast 2000 \
  --quantum-steps-slow 50000 \
  --report-every 50 \
  --seed 1 \
  --results-dir "$OUT" \
  --dist-m4ri-cmd dist_m4ri

# Equivalent:
# python -m qtanner.search --mode progressive <same args>

Notes:
- Default is to enumerate all multisets of size 6 containing the identity (no Cayley dedup unless `--dedup-cayley`).
- Ctrl-C stops after the current evaluation and prints the final best-by-k table.
- Outputs live under the run directory (classical JSONL + histograms, `best_codes/`, `milestones.jsonl`).
- Quantum distance evaluation runs a fast pass first; slow pass runs only if `d_fast >= ceil(sqrt(n))` and beats the current best-by-k bound for that k. Best-by-k entries are only recorded from the slow pass.

## 2) Monitor progress (macOS: no 'watch' needed)
OUT="$(ls -td runs/* | head -n 1)"

while true; do
  clear
  date
  echo "=== ${OUT}/best_by_k.txt ==="
  test -f "${OUT}/best_by_k.txt" && cat "${OUT}/best_by_k.txt" || echo "(missing)"
  echo
  echo "=== ${OUT}/coverage.txt ==="
  test -f "${OUT}/coverage.txt" && cat "${OUT}/coverage.txt" || echo "(missing)"
  sleep 2
done

## 2b) Monitor progress by tailing the log (works everywhere)
OUT="$(ls -td runs/* | head -n 1)"
tail -n 50 -F "${OUT}/best_by_k.log"

## 3) Re-check distance of a saved code (dist-m4ri RW)
# Replace ID with the code id you want to test.
OUT="$(ls -td runs/* | head -n 1)"
ID="PUT_CODE_ID_HERE"

python -m qtanner.check_distance \
  --run "$OUT" \
  --id "$ID" \
  --steps 5000 \
  --dist-m4ri-cmd dist_m4ri

## 4) Generate the LaTeX report for a run
OUT="$(ls -td runs/* | head -n 1)"
python -m qtanner.report --run "$OUT" --out "$OUT/report.tex"

# If you have pdflatex installed:
python -m qtanner.report --run "$OUT" --out "$OUT/report.tex" --pdf

## dist-m4ri install (short version)
- Build dist-m4ri and add the `dist_m4ri` binary to your `PATH`.
- Example (local checkout): `export PATH="$HOME/research/qtanner-tools/dist-m4ri/src:$PATH"`

## C2xC2xC2 target-16 run
- `scripts/run_c2xc2xc2_d16.sh`
- `scripts/run_progressive_c2xc2xc2_d16.sh`


## Groups of order < 20 (GAP SmallGroup identifiers)

### Order 1 (1 group(s))

- SmallGroup(1,1)

### Order 2 (1 group(s))

- SmallGroup(2,1)

### Order 3 (1 group(s))

- SmallGroup(3,1)

### Order 4 (2 group(s))

- SmallGroup(4,1)
- SmallGroup(4,2)

### Order 5 (1 group(s))

- SmallGroup(5,1)

### Order 6 (2 group(s))

- SmallGroup(6,1)
- SmallGroup(6,2)

### Order 7 (1 group(s))

- SmallGroup(7,1)

### Order 8 (5 group(s))

- SmallGroup(8,1)
- SmallGroup(8,2)
- SmallGroup(8,3)
- SmallGroup(8,4)
- SmallGroup(8,5)

### Order 9 (2 group(s))

- SmallGroup(9,1)
- SmallGroup(9,2)

### Order 10 (2 group(s))

- SmallGroup(10,1)
- SmallGroup(10,2)

### Order 11 (1 group(s))

- SmallGroup(11,1)

### Order 12 (5 group(s))

- SmallGroup(12,1)
- SmallGroup(12,2)
- SmallGroup(12,3)
- SmallGroup(12,4)
- SmallGroup(12,5)

### Order 13 (1 group(s))

- SmallGroup(13,1)

### Order 14 (2 group(s))

- SmallGroup(14,1)
- SmallGroup(14,2)

### Order 15 (1 group(s))

- SmallGroup(15,1)

### Order 16 (14 group(s))

- SmallGroup(16,1)
- SmallGroup(16,2)
- SmallGroup(16,3)
- SmallGroup(16,4)
- SmallGroup(16,5)
- SmallGroup(16,6)
- SmallGroup(16,7)
- SmallGroup(16,8)
- SmallGroup(16,9)
- SmallGroup(16,10)
- SmallGroup(16,11)
- SmallGroup(16,12)
- SmallGroup(16,13)
- SmallGroup(16,14)

### Order 17 (1 group(s))

- SmallGroup(17,1)

### Order 18 (5 group(s))

- SmallGroup(18,1)
- SmallGroup(18,2)
- SmallGroup(18,3)
- SmallGroup(18,4)
- SmallGroup(18,5)

### Order 19 (1 group(s))

- SmallGroup(19,1)
