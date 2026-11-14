#!/bin/bash
# Rehearsal matrix on the server: hidden-row RMSLE for several configurations and seeds.
#   bash scripts/reh_matrix.sh "0 1 2"   -> logs/reh_matrix.log (one summary line per run)
cd "$(dirname "$0")/.."
export PATH=$HOME/.local/bin:$PWD/.venv/bin:$PATH
SEEDS=${1:-"0 1 2"}
for seed in $SEEDS; do
  for cfg in full nog5 nomeas_nog5; do
    case $cfg in
      full) env_s=""; extra="";;
      nog5) env_s='EXTRA_STEPS=ident PREDICT_ARGS=--no-external'; extra="";;
      nomeas_nog5) env_s='EXTRA_STEPS=ident'; export PREDICT_ARGS="--measure none --no-external"; extra="--no-measure";;
    esac
    out=data_reh_tmp
    rm -rf "$out"
    if [ "$cfg" = nomeas_nog5 ]; then
      EXTRA_STEPS=ident python scripts/rehearsal.py --seed "$seed" --skip-phase-a --out "$out" $extra > logs/reh_${cfg}_${seed}.log 2>&1
      unset PREDICT_ARGS
    elif [ "$cfg" = nog5 ]; then
      EXTRA_STEPS=ident PREDICT_ARGS=--no-external python scripts/rehearsal.py --seed "$seed" --skip-phase-a --out "$out" > logs/reh_${cfg}_${seed}.log 2>&1
    else
      python scripts/rehearsal.py --seed "$seed" --skip-phase-a --out "$out" > logs/reh_${cfg}_${seed}.log 2>&1
    fi
    echo "$(date +%H:%M:%S) seed=$seed cfg=$cfg $(grep -E 'phase B \(graphs\)' logs/reh_${cfg}_${seed}.log | tail -1) | $(grep -E 'STACK\+G1' logs/reh_${cfg}_${seed}.log | tail -1 | sed 's/  */ /g')" >> logs/reh_matrix.log
  done
done
echo "MATRIX DONE" >> logs/reh_matrix.log
