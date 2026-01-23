#!/usr/bin/env bash
set -euo pipefail

OUT="${1:-scripts/groups_order_lt20.txt}"

gap -q -c '
for n in [1..19] do
  for i in [1..NumberSmallGroups(n)] do
    Print("SmallGroup(", n, ",", i, ")\n");
  od;
od;
quit;
' > "$OUT"

echo "[ok] wrote $OUT"
