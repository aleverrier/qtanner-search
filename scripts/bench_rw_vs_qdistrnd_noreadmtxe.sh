#!/usr/bin/env bash
set -euo pipefail

CODEDIR="${CODEDIR:-$HOME/research/qtanner-search/data/lrz_paper_mtx/633x633}"
DISTBIN="${DISTBIN:-$HOME/research/qtanner-tools/dist-m4ri/src/dist_m4ri}"
STEPS="${STEPS:-100000}"
WMIN="${WMIN:-0}"
SEED="${SEED:-1}"
GAPBIN="${GAPBIN:-/opt/homebrew/bin/gap}"

test -d "$CODEDIR"
test -x "$DISTBIN"
command -v "$GAPBIN" >/dev/null

TS="$(date +%Y%m%d_%H%M%S)"
OUTDIR="$HOME/research/qtanner-search/results/bench_rw_vs_qdistrnd_$TS"
mkdir -p "$OUTDIR"

echo "CODEDIR=$CODEDIR" | tee "$OUTDIR/runinfo.txt"
echo "DISTBIN=$DISTBIN" | tee -a "$OUTDIR/runinfo.txt"
echo "GAPBIN=$GAPBIN" | tee -a "$OUTDIR/runinfo.txt"
echo "STEPS=$STEPS WMIN=$WMIN SEED=$SEED" | tee -a "$OUTDIR/runinfo.txt"
echo "Started $(date)" | tee -a "$OUTDIR/runinfo.txt"

pick_codes_default() {
  ls -1 "$CODEDIR"/HX_*.mtx 2>/dev/null | head -n 2 | sed -e 's#.*/HX_##' -e 's/\.mtx$//'
}

CODES=()
if [ "$#" -ge 1 ]; then
  for a in "$@"; do CODES+=("$a"); done
else
  while IFS= read -r c; do
    [ -n "$c" ] && CODES+=("$c")
  done < <(pick_codes_default)
fi

if [ "${#CODES[@]}" -lt 1 ]; then
  echo "ERROR: No HX_*.mtx files found in $CODEDIR" >&2
  exit 1
fi

run_one_code() {
  local name="$1"
  local HX="$CODEDIR/HX_${name}.mtx"
  local HZ="$CODEDIR/HZ_${name}.mtx"

  if [ ! -f "$HX" ] || [ ! -f "$HZ" ]; then
    echo "Skipping $name (missing HX/HZ)" | tee -a "$OUTDIR/runinfo.txt"
    return
  fi

  echo "=== $name ===" | tee -a "$OUTDIR/runinfo.txt"

  local log_rw_z="$OUTDIR/${name}_distm4ri_RW_Z.txt"
  local log_rw_x="$OUTDIR/${name}_distm4ri_RW_X.txt"
  local log_gap="$OUTDIR/${name}_qdistrnd.txt"
  local tim="$OUTDIR/${name}_timing.txt"

  echo "dist-m4ri RW Z (steps=$STEPS)" | tee -a "$tim"
  /usr/bin/time -p -o /tmp/tim.$$ \
    "$DISTBIN" debug=0 method=1 finH="$HX" finG="$HZ" steps="$STEPS" wmin="$WMIN" seed="$SEED" \
    | tee "$log_rw_z" >/dev/null
  sed 's/^/  /' /tmp/tim.$$ | tee -a "$tim"

  echo "dist-m4ri RW X (steps=$STEPS)" | tee -a "$tim"
  /usr/bin/time -p -o /tmp/tim.$$ \
    "$DISTBIN" debug=0 method=1 finH="$HZ" finG="$HX" steps="$STEPS" wmin="$WMIN" seed="$SEED" \
    | tee "$log_rw_x" >/dev/null
  sed 's/^/  /' /tmp/tim.$$ | tee -a "$tim"

  cat > "$OUTDIR/_qdistrnd_tmp.g" <<GAPEOF
LoadPackage("QDistRnd");

ReadMMGF2 := function(path)
  local f, line, toks, dims, r, c, nz, M, i, j;

  f := InputTextFile(path);
  if f = fail then Error("Cannot open ", path); fi;

  # 1) header
  line := ReadLine(f);
  if line = fail then Error("Empty file: ", path); fi;

  # 2) skip comments (%...) until dims line
  repeat
    line := ReadLine(f);
    if line = fail then Error("Missing dims line: ", path); fi;
  until Length(line) = 0 or line[1] <> '%';

  toks := Filtered(SplitString(line, " \t\r\n", ""), x -> x <> "");
  if Length(toks) < 3 then Error("Bad dims line in ", path, ": ", line); fi;

  r := Int(toks[1]); c := Int(toks[2]); nz := Int(toks[3]);

  # build integer 0/1 matrix as list-of-lists, then coerce to GF(2)
  M := NullMat(r, c);

  while true do
    line := ReadLine(f);
    if line = fail then break; fi;
    if Length(line) = 0 then continue; fi;
    if line[1] = '%' then continue; fi;

    toks := Filtered(SplitString(line, " \t\r\n", ""), x -> x <> "");
    if Length(toks) = 2 then
      i := Int(toks[1]); j := Int(toks[2]);
      M[i][j] := 1;
    elif Length(toks) >= 3 then
      i := Int(toks[1]); j := Int(toks[2]);
      # interpret value mod 2
      if (Int(toks[3]) mod 2) <> 0 then M[i][j] := 1; fi;
    fi;
  od;

  CloseStream(f);
  return One(GF(2)) * M;
end;

HX := ReadMMGF2("$HX");
HZ := ReadMMGF2("$HZ");

Print("Code: $name\\n");

Print("DistRandCSS(HX,HZ,num=$STEPS,mindist=0,debug=0)\\n");
t0 := Runtime();
dz := DistRandCSS(HX, HZ, $STEPS, 0, 0 : field := GF(2));
t1 := Runtime();
Print("dZ = ", dz, "   time_ms=", (t1-t0), "\\n");

Print("DistRandCSS(HZ,HX,num=$STEPS,mindist=0,debug=0)\\n");
t0 := Runtime();
dx := DistRandCSS(HZ, HX, $STEPS, 0, 0 : field := GF(2));
t1 := Runtime();
Print("dX = ", dx, "   time_ms=", (t1-t0), "\\n");

QUIT;
GAPEOF

  echo "QDistRnd DistRandCSS (num=$STEPS) [custom MTX reader]" | tee -a "$tim"
  /usr/bin/time -p -o /tmp/tim.$$ "$GAPBIN" -q "$OUTDIR/_qdistrnd_tmp.g" | tee "$log_gap" >/dev/null
  sed 's/^/  /' /tmp/tim.$$ | tee -a "$tim"

  rm -f /tmp/tim.$$ "$OUTDIR/_qdistrnd_tmp.g"
}

for c in "${CODES[@]}"; do
  run_one_code "$c"
done

echo "Finished $(date)" | tee -a "$OUTDIR/runinfo.txt"
echo "Results in $OUTDIR"
