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
  --allow-repeats \
  --permH1 30 --permH1p 30 \
  --trials 2000 \
  --mindist 10 \
  --best-uniq-target 5 \
  --batch-size 200 \
  --seed 1 \
  --outdir "$OUT" \
  --gap-cmd gap

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

## 3) Re-check distance of a saved code (QDistRnd, mindist=0)
# Replace ID with the code id you want to test.
OUT="$(ls -td runs/* | head -n 1)"
ID="PUT_CODE_ID_HERE"

python -m qtanner.check_distance \
  --run "$OUT" \
  --id "$ID" \
  --trials 20000 \
  --uniq-target 5 \
  --gap-cmd gap

## 4) Generate the LaTeX report for a run
OUT="$(ls -td runs/* | head -n 1)"
python -m qtanner.report --run "$OUT" --out "$OUT/report.tex"

# If you have pdflatex installed:
python -m qtanner.report --run "$OUT" --out "$OUT/report.tex" --pdf


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

