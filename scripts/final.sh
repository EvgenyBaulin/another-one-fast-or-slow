#!/bin/bash
# Last-hour pipeline: new ONNX files appear on Kaggle -> final submission CSV. NEVER submits.
#
#   scripts/final.sh [--wait] [--local] [--no-measure] [--measure-cap SEC] [--tag NAME]
#
# Runs on the Mac, safe to re-run at any moment (a re-run takes over a server job left by
# a previous run; downloads resume; measurement is incremental).
#   step 0  check the Kaggle listing for files we do not have (--wait: poll every 30 s)
#   step 1  server: refresh the 3 CSVs (.bak + loud message if train/test changed), sync files
#           (fetch_data.py re-run under a timeout while something is missing, total budget)
#   step 2  features.py   step 3  measure.py cuda, then cpu, test rows only, wall-clock cap
#   step 3b g5.py (extra stack members, they go stale when new graphs arrive), capped
#   step 4  predict.py --blind prior -> submissions/final_<HHMM>.csv
#   step 5  pull the CSV to the Mac, validate, print the submit command for the main session
# Server jobs run detached (nohup) and are streamed, so a dropped ssh does not kill them.
# --local: everything on the Mac (.venv/bin/python, data/), steps 1-2-4, no measurement.
#
# Tunables (env): FETCH_BUDGET=1200 FETCH_ROUND=240 FETCH_JOBS=8 MEASURE_CAP=600
#   MEASURE_CUDA_SHARE=50 (% of the cap cuda may use)  MEASURE_SCOPE=test|all
#   MEASURE_FORCE=1 (measure a provider even if data/measure_<p>.csv does not exist yet)
#   MEASURE_CUDA_ARGS='--gpus 0,1 --timeout 120'  MEASURE_CPU_ARGS='--workers 4 --threads 4 --timeout 120'
#   PREDICT_ARGS (extra args for predict.py; no quotes inside any of the three)
#   EXTRA_STEPS=g5 EXTRA_CAP=300 EXTRA_ARGS= (scripts/g5.py --data D refreshes <data>/preds; EXTRA_STEPS= disables)
#
# Internal modes (used on the server and by scripts/rehearsal.py, same code as the real run):
#   final.sh __check --data D | __fetch --data D | __pipeline --data D --out CSV [--no-measure]
set -uo pipefail
{  # one block: bash reads all of it before executing
cd "$(dirname "$0")/.."

REMOTE=team06
RDIR=hacklab/fos
SSH="ssh -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=10 -o ServerAliveCountMax=3 -o LogLevel=ERROR"
FETCH_BUDGET=${FETCH_BUDGET:-1200}
FETCH_ROUND=${FETCH_ROUND:-240}
FETCH_JOBS=${FETCH_JOBS:-8}
MEASURE_CAP=${MEASURE_CAP:-600}
MEASURE_CUDA_SHARE=${MEASURE_CUDA_SHARE:-50}
MEASURE_SCOPE=${MEASURE_SCOPE:-test}
MEASURE_FORCE=${MEASURE_FORCE:-0}
MEASURE=${MEASURE:-1}
EXTRA_STEPS=${EXTRA_STEPS-feat2 base2 ident g6}   # scripts/<name>.py --data D, run between measure and predict
EXTRA_CAP=${EXTRA_CAP:-420}
# must match how train was measured (main session: cuda --gpus 0, cpu --workers 4 --threads 4)
MEASURE_CUDA_ARGS=${MEASURE_CUDA_ARGS:---gpus 0 --timeout 120}
MEASURE_CPU_ARGS=${MEASURE_CPU_ARGS:---workers 4 --threads 4 --timeout 120}
if [ -x .venv/bin/python ]; then PY="$PWD/.venv/bin/python"; else PY=python3; fi
if [ "$(uname)" = Darwin ]; then JOBS=${JOBS:-8}; else JOBS=${JOBS:-20}; fi

say() { echo "[final] $*"; }
loud() { echo; echo "!!!!!!!!!! $* !!!!!!!!!!"; echo; }

# run_capped <seconds> <INT|TERM> <cmd...>: own process group, signal the whole group on
# timeout, KILL after 30 s; exit code 124 on timeout (works on Linux and macOS).
run_capped() {
  "$PY" -c '
import os, signal, subprocess, sys
cap, sig, cmd = float(sys.argv[1]), getattr(signal, "SIG" + sys.argv[2]), sys.argv[3:]
# a detached job inherits SIGINT=ignore; reset it so measure.py can stop gracefully
p = subprocess.Popen(cmd, process_group=0, preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_DFL))
def group(s):
    try:
        os.killpg(p.pid, s)
    except (ProcessLookupError, PermissionError):
        pass
signal.signal(signal.SIGTERM, lambda *_: (group(signal.SIGKILL), sys.exit(143)))
try:
    rc = p.wait(timeout=cap)
    group(signal.SIGKILL)
    sys.exit(rc)
except subprocess.TimeoutExpired:
    pass
group(sig)
try:
    p.wait(timeout=30)
except subprocess.TimeoutExpired:
    pass
group(signal.SIGKILL)
p.wait()
sys.exit(124)
' "$@"
}

# ---------------------------------------------------------------- step 0 (where the data is)
do_check() {  # prints one line: CHECK listing=.. to_fetch=.. to_fetch_onnx=.. ...
  "$PY" - "$1" <<'PYEOF'
import csv, os, sys
from pathlib import Path
sys.path.insert(0, "scripts")
import fetch_data as F
out = Path(sys.argv[1])
files = F.list_files(F.token())
todo = [f for f in files if not (F.local_path(out, f["name"]).exists()
        and F.local_path(out, f["name"]).stat().st_size == f["totalBytes"])]
listed = {os.path.basename(f["name"]) for f in files}
have = set(os.listdir(out / "data")) if (out / "data").is_dir() else set()
tfiles = [r["file"] for r in csv.DictReader(open(out / "test.csv"))] if (out / "test.csv").exists() else []
print(f"CHECK listing={len(files)} listing_onnx={sum(f['name'].endswith('.onnx') for f in files)} "
      f"to_fetch={len(todo)} to_fetch_onnx={sum(f['name'].endswith('.onnx') for f in todo)} "
      f"to_fetch_mb={sum(f['totalBytes'] for f in todo) / 1e6:.0f} "
      f"test_rows_not_listed={sum(f not in listed for f in tfiles)} "
      f"test_rows_no_file={sum(f not in have for f in tfiles)}")
PYEOF
}

# ---------------------------------------------------------------- step 1 (where the data is)
do_fetch() {
  local data=$1 t0=$SECONDS round=0 bad=0 left cap rc chk
  # 1a. CSVs first, into a staging dir: fetch_data.py resumes by size and would corrupt a
  # changed CSV; after this their sizes match the listing and the main sync skips them.
  "$PY" - "$data" <<'PYEOF'
import csv, filecmp, os, shutil, sys, time
from pathlib import Path
sys.path.insert(0, "scripts")
import fetch_data as F
out = Path(sys.argv[1])
tok = F.token()
stage = out / ".csv_staging"
shutil.rmtree(stage, ignore_errors=True)
stage.mkdir(parents=True)
def ids(p):
    with open(p, newline="") as f:
        return [r.get("id") for r in csv.DictReader(f)]
for f in F.list_files(tok):
    n = f["name"]
    if not n.endswith(".csv"):
        continue
    new, cur = F.local_path(stage, n), F.local_path(out, n)
    msg = ""
    for _ in range(3):
        new.unlink(missing_ok=True)
        msg = F.download(tok, stage, n, f["totalBytes"])
        if msg.startswith("ok"):
            break
    if not msg.startswith("ok") or "id" not in open(new).readline():
        print(f"\n!!!!!!!!!! CSV REFRESH FAILED for {n}: {msg} -- keeping the current copy !!!!!!!!!!\n", flush=True)
        continue
    if not cur.exists():
        cur.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(new, cur)
        print(f"csv {n}: new file", flush=True)
    elif filecmp.cmp(new, cur, shallow=False):
        print(f"csv {n}: unchanged", flush=True)
    else:
        bak = cur.with_name(cur.name + ".bak")
        if bak.exists():
            bak = cur.with_name(cur.name + time.strftime(".bak.%H%M%S"))
        shutil.copy2(cur, bak)
        a, b = ids(cur), ids(new)
        os.replace(new, cur)
        print(f"\n!!!!!!!!!! {n} CHANGED ON KAGGLE: rows {len(a)} -> {len(b)}, ids added {len(set(b) - set(a))}, "
              f"removed {len(set(a) - set(b))}, order changed {a != b}; old copy kept as {bak.name} !!!!!!!!!!\n", flush=True)
shutil.rmtree(stage, ignore_errors=True)
PYEOF
  rc=$?; [ $rc -ne 0 ] && loud "CSV refresh step failed (rc=$rc), continuing with the current CSVs"
  # 1b. sync files, re-running the fetch while something is missing, under a total budget
  while :; do
    left=$((FETCH_BUDGET - (SECONDS - t0)))
    chk=$("$PY" scripts/fetch_data.py --out "$data" --check 2>&1); rc=$?
    echo "$chk" | grep '^listing:' || echo "$chk" | tail -n 2
    if [ $rc -eq 0 ]; then say "fetch: nothing missing"; break; fi
    if ! echo "$chk" | grep -q '^listing:'; then
      bad=$((bad + 1))
      if [ $bad -ge 3 ]; then loud "Kaggle listing failed 3 times (token/network?) -- going on with the files we have"; break; fi
      sleep 5; continue
    fi
    if [ $left -le 10 ]; then loud "fetch budget ${FETCH_BUDGET}s is used up, files still missing -- going on; re-run final.sh to get the rest"; break; fi
    round=$((round + 1)); cap=$((left < FETCH_ROUND ? left : FETCH_ROUND))
    say "fetch round $round (timeout ${cap}s, budget left ${left}s)"
    run_capped "$cap" TERM "$PY" scripts/fetch_data.py --out "$data" --jobs "$FETCH_JOBS" 2>&1 | grep -v '^listing:'
    rc=${PIPESTATUS[0]}
    [ $rc -eq 124 ] && say "fetch round $round hit the ${cap}s timeout, restarting (downloads resume)"
  done
  # 1c. files that arrived under another path or inside a zip: link them into <data>/data
  "$PY" - "$data" <<'PYEOF'
import csv, os, sys, zipfile
from pathlib import Path
out = Path(sys.argv[1])
need = set()
for s in ("train", "test"):
    if (out / f"{s}.csv").exists():
        need |= {r["file"] for r in csv.DictReader(open(out / f"{s}.csv"))}
dd = out / "data"
missing = need - set(os.listdir(dd))
if missing:
    for z in out.rglob("*.zip"):
        tgt = out / "_unzipped" / z.stem
        if not tgt.exists():
            try:
                zipfile.ZipFile(z).extractall(tgt)
                print(f"unzipped {z} -> {tgt}", flush=True)
            except Exception as e:
                print(f"!!!!!!!!!! cannot unzip {z}: {e}", flush=True)
    found = {p.name: p for p in out.rglob("*.onnx") if p.parent != dd}
    for f in sorted(missing & found.keys()):
        os.symlink(found[f].resolve(), dd / f)
    if missing & found.keys():
        print(f"\n!!!!!!!!!! {len(missing & found.keys())} files were delivered outside data/data, linked into {dd} !!!!!!!!!!\n")
tfiles = [r["file"] for r in csv.DictReader(open(out / "test.csv"))]
have = set(os.listdir(dd))
print(f"FETCH onnx_present={len(have)} test_rows={len(tfiles)} test_rows_no_file={sum(f not in have for f in tfiles)}")
PYEOF
  say "step 1 fetch: $((SECONDS - t0))s"
}

# ---------------------------------------------------------------- steps 2-4 (where the data is)
do_pipeline() {
  local data=$1 out=$2 t tp=$SECONDS rc nog ids help prov left cap extra why x
  local -a margs
  # step 2
  t=$SECONDS
  "$PY" scripts/features.py --data "$data" --out "$data/features.csv" --jobs "$JOBS" 2>&1 | tail -n 4; rc=${PIPESTATUS[0]}
  if [ $rc -ne 0 ]; then
    say "features failed (rc=$rc), one retry"; sleep 3
    "$PY" scripts/features.py --data "$data" --out "$data/features.csv" --jobs "$JOBS" 2>&1 | tail -n 15; rc=${PIPESTATUS[0]}
  fi
  if [ $rc -ne 0 ]; then
    if [ -s "$data/features.csv" ]; then loud "features.py FAILED twice, using the STALE $data/features.csv"
    else loud "features.py FAILED and there is no features.csv -- stop"; return 2; fi
  fi
  say "step 2 features: $((SECONDS - t))s rc=$rc"
  nog=$("$PY" -c '
import csv, os, sys
d = sys.argv[1]
have = set(os.listdir(os.path.join(d, "data")))
rows = list(csv.DictReader(open(os.path.join(d, "test.csv"))))
print(sum(r["file"] not in have for r in rows))
print(",".join(r["id"] for r in rows if r["file"] in have))' "$data")
  ids=$(echo "$nog" | sed -n 2p); nog=$(echo "$nog" | sed -n 1p)
  say "test rows without a graph: $nog"
  # step 3
  t=$SECONDS
  if [ "$MEASURE" != 1 ]; then say "step 3 measure: skipped (--no-measure)"
  elif [ ! -f scripts/measure.py ]; then say "step 3 measure: skipped (scripts/measure.py is absent)"
  elif ! help=$("$PY" scripts/measure.py --help 2>&1); then say "step 3 measure: skipped (measure.py --help fails: $(echo "$help" | tail -n 1))"
  else
    for prov in cuda cpu; do
      left=$((MEASURE_CAP - (SECONDS - t)))
      if [ ! -s "$data/measure_$prov.csv" ] && [ "$MEASURE_FORCE" != 1 ]; then
        say "step 3 measure $prov: skipped (no $data/measure_$prov.csv, i.e. train was never measured; MEASURE_FORCE=1 overrides)"; continue
      fi
      if [ $left -lt 15 ]; then say "step 3 measure $prov: skipped (cap ${MEASURE_CAP}s is used up)"; continue; fi
      if [ $prov = cuda ]; then cap=$((MEASURE_CAP * MEASURE_CUDA_SHARE / 100)); [ $cap -gt $left ] && cap=$left; extra=$MEASURE_CUDA_ARGS
      else cap=$left; extra=$MEASURE_CPU_ARGS; fi
      margs=(--data "$data" --provider "$prov" --out "$data/measure_$prov.csv")
      if [ "$MEASURE_SCOPE" = test ] && [ -n "$ids" ]; then
        case "$help" in *--ids*) margs+=(--ids "$ids") ;; *) say "measure.py has no --ids, measuring everything" ;; esac
      fi
      local tm=$SECONDS try=0 capleft
      while :; do  # a crash (not a timeout) is retried: measure.py resumes from its journal
        try=$((try + 1)); capleft=$((cap - (SECONDS - tm)))
        # shellcheck disable=SC2086
        run_capped "$capleft" INT "$PY" scripts/measure.py "${margs[@]}" $extra 2>&1 | grep -Ev '^\[[0-9]+/[0-9]+\]' | tail -n 6
        rc=${PIPESTATUS[0]}
        if [ $rc -eq 0 ] || [ $rc -eq 124 ] || [ $try -ge 3 ] || [ $((cap - (SECONDS - tm))) -lt 20 ]; then break; fi
        say "measure $prov crashed (rc=$rc), re-running to finish the pending rows (try $((try + 1)) of 3)"
      done
      case $rc in 0) why=ok ;; 124) why="TIMEOUT at ${cap}s, partial results kept" ;; *) why="FAILED, continuing without it" ;; esac
      if [ $rc -ne 0 ]; then  # make sure the CSV reflects the journal even after a hard stop
        case "$help" in *--limit*) run_capped 60 TERM "$PY" scripts/measure.py --data "$data" --provider "$prov" --out "$data/measure_$prov.csv" --limit 0 2>&1 | tail -n 1 ;; esac
      fi
      say "step 3 measure $prov: $((SECONDS - tm))s rc=$rc ($why)"
    done
    say "step 3 measure total: $((SECONDS - t))s (cap ${MEASURE_CAP}s)"
  fi
  # step 3b: extra ensemble members that predict.py reads from <data>/preds (G5). They only
  # cover test rows that had a graph when they were computed, so after new files arrive they
  # are stale and predict.py silently drops them unless they are recomputed here.
  for x in $EXTRA_STEPS; do
    if [ ! -f "scripts/$x.py" ]; then say "step 3b $x: skipped (scripts/$x.py is absent)"; continue; fi
    t=$SECONDS
    # shellcheck disable=SC2086
    run_capped "$EXTRA_CAP" TERM "$PY" "scripts/$x.py" --data "$data" ${EXTRA_ARGS:-} 2>&1 | tail -n 4; rc=${PIPESTATUS[0]}
    if [ $rc -eq 0 ]; then why=ok; else why="FAILED or timeout ${EXTRA_CAP}s: predict.py will go without this model"; fi
    say "step 3b $x: $((SECONDS - t))s rc=$rc ($why)"
  done
  # step 4
  t=$SECONDS
  mkdir -p "$(dirname "$out")"; rm -f "$out"
  # shellcheck disable=SC2086
  "$PY" scripts/predict.py --data "$data" --out "$out" --blind prior ${PREDICT_ARGS:-} 2>&1 | tail -n 45; rc=${PIPESTATUS[0]}
  if [ $rc -ne 0 ] || [ ! -s "$out" ]; then
    say "predict failed (rc=$rc), one retry"; sleep 3
    # shellcheck disable=SC2086
    "$PY" scripts/predict.py --data "$data" --out "$out" --blind prior ${PREDICT_ARGS:-} 2>&1 | tail -n 30; rc=${PIPESTATUS[0]}
  fi
  say "step 4 predict: $((SECONDS - t))s rc=$rc"
  if [ $rc -ne 0 ] || [ ! -s "$out" ]; then loud "predict.py FAILED -- no submission file"; return 4; fi
  say "pipeline done: out=$out no_graph=$nog total=$((SECONDS - tp))s"
}

# ---------------------------------------------------------------- internal modes
if [ "${1:-}" = __check ] || [ "${1:-}" = __fetch ] || [ "${1:-}" = __pipeline ]; then
  mode=$1; shift; D=data; O=""
  while [ $# -gt 0 ]; do
    case $1 in
      --data) D=$2; shift 2 ;;
      --out) O=$2; shift 2 ;;
      --no-measure) MEASURE=0; shift ;;
      --measure-cap) MEASURE_CAP=$2; shift 2 ;;
      *) echo "unknown arg $1" >&2; exit 64 ;;
    esac
  done
  case $mode in
    __check) do_check "$D" ;;
    __fetch) do_fetch "$D" ;;
    __pipeline) [ -n "$O" ] || { echo "--out is required" >&2; exit 64; }; do_pipeline "$D" "$O" ;;
  esac
  exit $?
fi

# ---------------------------------------------------------------- Mac side
WAIT=0; LOCAL=0; TAG=""
while [ $# -gt 0 ]; do
  case $1 in
    --wait) WAIT=1; shift ;;
    --local) LOCAL=1; shift ;;
    --no-measure) MEASURE=0; shift ;;
    --measure-cap) MEASURE_CAP=$2; shift 2 ;;
    --tag) TAG=$2; shift 2 ;;
    -h | --help) sed -n 2,25p "$0"; exit 0 ;;
    *) echo "unknown arg $1 (see --help)" >&2; exit 64 ;;
  esac
done
T0=$SECONDS
TMPD=$(mktemp -d "${TMPDIR:-/tmp}/final.XXXXXX")
trap 'rm -rf "$TMPD"' EXIT
OUT="submissions/final_${TAG:-$(TZ=Europe/Moscow date +%H%M)}.csv"
TOKEN_FILE="$HOME/.kaggle/access_token"
[ -s "$TOKEN_FILE" ] || { loud "no Kaggle token in $TOKEN_FILE"; exit 2; }
token_line() { tr -d '\r\n' < "$TOKEN_FILE"; printf '\n'; }
rsh() { $SSH "$REMOTE" "cd ~/$RDIR && export PATH=\$HOME/.local/bin:\$PWD/.venv/bin:\$PATH && { $*
}"; }
sync_code() {
  local i
  for i in 1 2 3; do scripts/remote.sh true < /dev/null && return 0; sleep 5; done
  return 1
}

# remote_job <name> <command>: start the command detached on the server (stdin line 1 becomes
# KAGGLE_API_TOKEN in its environment: never argv, never disk), stream its log and return its
# exit code. A dropped ssh only interrupts the streaming; a re-run kills the previous job.
remote_job() {
  local name=$1 cmd=$2 rlog="logs/final_$1.log" sid="logs/final_$1.sid" llog="$TMPD/job_$1.log" n drops=0 rc cur jid="$$-$RANDOM"
  : > "$llog"
  rsh "mkdir -p logs; IFS= read -r KAGGLE_API_TOKEN || true; export KAGGLE_API_TOKEN
echo __JOB=$jid > $rlog
S=\$(cat $sid 2>/dev/null || true)
if [ -n \"\$S\" ] && pgrep -s \"\$S\" -f final.sh > /dev/null 2>&1; then
  echo '[final] taking over a previous $name job on the server'; pkill -TERM -s \"\$S\"; sleep 2; pkill -KILL -s \"\$S\"
fi
setsid nohup bash -c 'echo \$\$ > $sid; $cmd; rc=\$?; rm -f $sid; echo __RC=\$rc' >> $rlog 2>&1 < /dev/null &
sleep 1" || { say "cannot start the $name job on the server"; return 255; }
  while :; do
    n=$(wc -l < "$llog" | tr -d ' ')
    rsh "if [ -s $sid ]; then tail -n +$((n + 1)) -f --pid=\$(cat $sid) $rlog; else tail -n +$((n + 1)) $rlog; fi" < /dev/null \
      | tee -a "$llog" | grep --line-buffered -Ev '^__(RC|JOB)=' || true
    cur=$(rsh "head -n 1 $rlog" < /dev/null 2> /dev/null)
    if [ -n "$cur" ] && [ "$cur" != "__JOB=$jid" ]; then
      loud "this run was superseded by a newer scripts/final.sh (its $name job replaced ours) -- stopping"; return 7
    fi
    grep -q '^__RC=' "$llog" && break
    drops=$((drops + 1))
    if [ $drops -gt 60 ]; then loud "lost the server for too long; the $name job may still run there, re-run final.sh"; return 255; fi
    say "ssh dropped ($drops), the $name job keeps running on the server, reconnecting"; sleep 4
  done
  rc=$(sed -n 's/^__RC=//p' "$llog" | tail -n 1)
  return "${rc:-1}"
}

if [ $LOCAL = 1 ]; then
  say "LOCAL mode: everything on the Mac, no measurement"
  KAGGLE_API_TOKEN=$(tr -d '\r\n' < "$TOKEN_FILE"); export KAGGLE_API_TOKEN
  MEASURE=0
  [ -n "${EXTRA_STEPS_LOCAL:-}" ] || EXTRA_STEPS=""   # emergency mode stays minimal (no TabPFN on the Mac)
  check() { do_check data; }
else
  if ! sync_code; then
    loud "server $REMOTE is unreachable (3 tries). Fallback: scripts/final.sh --local"
    exit 3
  fi
  check() { token_line | rsh "IFS= read -r KAGGLE_API_TOKEN; export KAGGLE_API_TOKEN; bash scripts/final.sh __check --data data"; }
fi

# step 0
t=$SECONDS; poll=0
while :; do
  poll=$((poll + 1))
  C=$(check 2>&1 | grep '^CHECK ' | tail -n 1)
  NEW=$(echo "$C" | sed -n 's/.* to_fetch=\([0-9]*\).*/\1/p')
  if [ -z "$NEW" ]; then say "$(date +%H:%M:%S) listing check failed (network/token?)"; [ $WAIT = 1 ] || break
  elif [ "$NEW" -gt 0 ]; then say "NEW FILES in the Kaggle listing: $C"; break
  elif [ $WAIT = 0 ]; then say "nothing new in the Kaggle listing: $C"; break
  else say "$(date +%H:%M:%S) waiting, nothing new yet (poll $poll): ${C#CHECK }"; fi
  sleep 30
done
say "step 0 check: $((SECONDS - t))s"

# steps 1-4
t=$SECONDS
if [ $LOCAL = 1 ]; then
  do_fetch data
  do_pipeline data "$OUT"; RC=$?
  NOG=$("$PY" -c 'import csv, os; h = set(os.listdir("data/data")); print(sum(r["file"] not in h for r in csv.DictReader(open("data/test.csv"))))')
  SAMPLE=data/sample_submission.csv
else
  token_line | remote_job fetch "FETCH_BUDGET=$FETCH_BUDGET FETCH_ROUND=$FETCH_ROUND FETCH_JOBS=$FETCH_JOBS bash scripts/final.sh __fetch --data data"
  FRC=$?; say "step 1 (with ssh): $((SECONDS - t))s rc=$FRC"
  [ $FRC -eq 7 ] && exit 7
  sync_code || say "code re-sync failed, the server keeps the scripts from the start of this run"
  t=$SECONDS
  remote_job pipeline "EXTRA_STEPS=\"$EXTRA_STEPS\" EXTRA_ARGS=\"${EXTRA_ARGS:-}\" EXTRA_CAP=$EXTRA_CAP MEASURE=$MEASURE MEASURE_CAP=$MEASURE_CAP MEASURE_CUDA_SHARE=$MEASURE_CUDA_SHARE MEASURE_SCOPE=$MEASURE_SCOPE MEASURE_FORCE=$MEASURE_FORCE MEASURE_CUDA_ARGS=\"$MEASURE_CUDA_ARGS\" MEASURE_CPU_ARGS=\"$MEASURE_CPU_ARGS\" PREDICT_ARGS=\"${PREDICT_ARGS:-}\" bash scripts/final.sh __pipeline --data data --out $OUT" < /dev/null
  RC=$?; [ $RC -eq 7 ] && exit 7
  say "steps 2-4 (with ssh): $((SECONDS - t))s rc=$RC"
  NOG=$(sed -n 's/.*pipeline done: .* no_graph=\([0-9]*\).*/\1/p' "$TMPD/job_pipeline.log" | tail -n 1)
fi
if [ $RC -ne 0 ]; then loud "PIPELINE FAILED (rc=$RC): no new submission. Fix and re-run scripts/final.sh"; exit 4; fi

# step 5
t=$SECONDS
if [ $LOCAL = 0 ]; then
  rm -f "$OUT"
  for i in 1 2 3; do
    rsync -az -e "$SSH" "$REMOTE:$RDIR/$OUT" "$OUT" \
      && rsync -az -e "$SSH" "$REMOTE:$RDIR/data/train.csv" "$REMOTE:$RDIR/data/test.csv" "$REMOTE:$RDIR/data/sample_submission.csv" "$TMPD/" && break
    sleep 4
  done
  [ -s "$OUT" ] || { loud "cannot pull $OUT from the server; it is there: ~/$RDIR/$OUT"; exit 5; }
  SAMPLE="$TMPD/sample_submission.csv"
  for f in train.csv test.csv sample_submission.csv; do  # keep the Mac copies equal (submit.sh checks against data/)
    if [ -s "$TMPD/$f" ] && ! cmp -s "$TMPD/$f" "data/$f"; then
      [ -e "data/$f" ] && cp -p "data/$f" "data/$f.bak.$(date +%H%M%S)"
      cp "$TMPD/$f" "data/$f"
      loud "data/$f on the Mac differed from the server copy: replaced, old one kept as .bak.*"
    fi
  done
fi
"$PY" -I - "$OUT" "$SAMPLE" "${NOG:-unknown}" <<'PYEOF'
import sys
import numpy as np, pandas as pd
s, ref, nog = pd.read_csv(sys.argv[1]), pd.read_csv(sys.argv[2]), sys.argv[3]
errs = []
if list(s.columns) != ["id", "latency"]: errs.append(f"columns {list(s.columns)}")
if len(s) != len(ref): errs.append(f"{len(s)} rows, sample has {len(ref)}")
elif not (s.id.values == ref.id.values).all(): errs.append("ids differ from the sample order")
v = pd.to_numeric(s.latency, errors="coerce").values if "latency" in s else np.array([np.nan])
if not np.isfinite(v).all(): errs.append(f"{int((~np.isfinite(v)).sum())} non-finite values")
if (v[np.isfinite(v)] < 0).any(): errs.append("negative values")
if len(ref) != 299: print(f"!!!!!!!!!! sample_submission has {len(ref)} rows, expected 299 (test changed?)")
if (v == 0).any(): print(f"!!!!!!!!!! {int((v == 0).sum())} rows are exactly 0")
print(f"[final] validate: rows={len(s)} min={np.nanmin(v):.4g} median={np.nanmedian(v):.4g} max={np.nanmax(v):.4g} "
      f"| test rows still without a graph: {nog}")
if errs:
    print("!!!!!!!!!! INVALID SUBMISSION: " + "; ".join(errs))
    sys.exit(1)
PYEOF
VRC=$?
say "step 5 pull+validate: $((SECONDS - t))s rc=$VRC"
say "total: $((SECONDS - T0))s"
if [ $VRC -ne 0 ]; then loud "DO NOT SUBMIT $OUT"; exit 6; fi
[ "${NOG:-0}" != 0 ] && say "NOTE: ${NOG} test rows have no graph yet (batch prior used). Re-run scripts/final.sh when the files arrive."
echo
say "READY: $OUT -- nothing was submitted. Main session submits with:"
echo "  scripts/submit.sh $OUT \"$(basename "$OUT" .csv), no-graph rows: ${NOG:-?}\""
exit 0
}
