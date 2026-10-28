#!/usr/bin/env bash
# Sequential CPU analyses (one process at a time: the server has 15 GB RAM and no swap).
set -uo pipefail
for args in "--task bootstrap --kind last:band+mean:0.75+mean:1 --C 0.1" "--task sweep --C 0.1" "--task perm --kind last:band+mean:0.75+mean:1 --C 0.1 --perms 200"; do
  echo "== $args"; bash run_analysis.sh $args
done
echo ALL_DONE
