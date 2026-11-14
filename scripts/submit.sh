#!/bin/bash
# Submit a CSV to the competition from the Mac and print the resulting score.
#   scripts/submit.sh submissions/v0.csv "message"
# Uses the Kaggle CLI through uvx; the token is read from ~/.kaggle/access_token.
set -euo pipefail
cd "$(dirname "$0")/.."
C=another-one-fast-or-slow-predict-ai-model-runtime-again
F="$1"; MSG="${2:-$(basename "$F")}"
.venv/bin/python -I - "$F" data/sample_submission.csv <<'PY'
import sys, pandas as pd, numpy as np
s, ref = pd.read_csv(sys.argv[1]), pd.read_csv(sys.argv[2])
assert list(s.columns) == ["id", "latency"], s.columns
assert len(s) == len(ref) and (s.id.values == ref.id.values).all(), "ids differ from sample"
assert s.latency.notna().all() and np.isfinite(s.latency).all() and (s.latency >= 0).all(), "bad values"
print(f"check ok: {len(s)} rows, zeros: {(s.latency == 0).sum()}")
PY
export KAGGLE_API_TOKEN="$(cat ~/.kaggle/access_token)"
uvx --with six --with python-dateutil kaggle competitions submit -c "$C" -f "$F" -m "$MSG" 2>&1 | tail -2
for i in 1 2 3 4 5 6; do
  sleep 5
  OUT=$(uvx --with six --with python-dateutil kaggle competitions submissions -c "$C" --csv 2>/dev/null | head -2 | tail -1)
  echo "$OUT" | grep -qi "pending" || break
done
echo "$OUT"
echo "$(date -u +%FT%TZ) | $F | $MSG | $OUT" >> submissions/log.txt
