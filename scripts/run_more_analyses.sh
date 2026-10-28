#!/usr/bin/env bash
# Runs after run_all_analyses.sh finishes (one CPU process at a time).
set -uo pipefail
while pgrep -f run_all_analyses.sh > /dev/null; do sleep 20; done
for args in "--task bootstrap --eval-all --kind last:band+mean:0.75+mean:1 --C 0.1" "--task weights --kind last:band+mean:0.75+mean:1 --C 0.1"; do
  echo "== $args"; bash run_analysis.sh $args
done
echo MORE_DONE
