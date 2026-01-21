# qtanner search run commands

## Command used
`/Users/leverrie/research/.venv/bin/python /Users/leverrie/research/qtanner-search/src/qtanner/search.py --groups C2xC2xC2 --max-n 288 --steps 4000 --target-distance 16 --batch-size 150 --frontier-max-per-point 100 --frontier-max-total 200 --max-quantum 500 --A-enum multiset --B-enum multiset --seed 1 --outdir results/c2xc2xc2_d16_20260120_150123 --dist-m4ri-cmd dist_m4ri --steps 5000 --target-distance 16 --seed 1768922122 --A-enum multiset --B-enum multiset`

## Reproducible module form
`python -m qtanner.search --groups C2xC2xC2 --max-n 288 --steps 4000 --target-distance 16 --batch-size 150 --frontier-max-per-point 100 --frontier-max-total 200 --max-quantum 500 --A-enum multiset --B-enum multiset --seed 1 --outdir results/c2xc2xc2_d16_20260120_150123 --dist-m4ri-cmd dist_m4ri --steps 5000 --target-distance 16 --seed 1768922122 --A-enum multiset --B-enum multiset`

## Raw argv
`["/Users/leverrie/research/qtanner-search/src/qtanner/search.py", "--groups", "C2xC2xC2", "--max-n", "288", "--steps", "4000", "--target-distance", "16", "--batch-size", "150", "--frontier-max-per-point", "100", "--frontier-max-total", "200", "--max-quantum", "500", "--A-enum", "multiset", "--B-enum", "multiset", "--seed", "1", "--outdir", "results/c2xc2xc2_d16_20260120_150123", "--dist-m4ri-cmd", "dist_m4ri", "--steps", "5000", "--target-distance", "16", "--seed", "1768922122", "--A-enum", "multiset", "--B-enum", "multiset"]`

## Monitor progress
OUT="results/c2xc2xc2_d16_20260120_150123"
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

## Monitor by tailing the log
tail -n 50 -F "${OUT}/best_by_k.log"

## Re-check distance (dist-m4ri RW)
ID="PUT_CODE_ID_HERE"
python -m qtanner.check_distance --run "${OUT}" --id "${ID}" --steps 5000 --dist-m4ri-cmd dist_m4ri --target-distance 16

Example:
ID="EXAMPLE_CODE_ID"
python -m qtanner.check_distance --run "${OUT}" --id "${ID}" --steps 5000 --dist-m4ri-cmd dist_m4ri --target-distance 16

## Generate LaTeX report
python -m qtanner.report --run "${OUT}" --out "${OUT}/report.tex"

Optional PDF:
python -m qtanner.report --run "${OUT}" --out "${OUT}/report.tex" --pdf
