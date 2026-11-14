#!/bin/bash
# Validation of --exact-shapes: same predict / g1 commands on two symlinked copies of data/.
# data_base = current features.csv, data_exact = data/features_exact.csv. Own preds/ and base2_feats.csv
# so nothing shared in data/ is written.
set -u
mkdir -p /tmp/ex
for d in data_base data_exact; do
  rm -rf "$d"; mkdir -p "$d/preds"
  for f in data/*; do
    b=$(basename "$f")
    case "$b" in features.csv|preds|base2_feats.csv|features_exact.csv|exact_shapes|base2_nodes.pkl) ;; *) ln -sfn "$PWD/data/$b" "$d/$b" ;; esac
  done
done
cp data/features.csv data_base/features.csv
cp data/features_exact.csv data_exact/features.csv
sleep 1
for d in data_base data_exact; do cp data/base2_feats.csv "$d/base2_feats.csv"; touch "$d/base2_feats.csv"; done
for d in data_base data_exact; do
  nohup timeout 345 python scripts/predict.py --data "$d" --no-g1 --g6-weight 0 --out "/tmp/ex/pred_$d.csv" > "/tmp/ex/pred_$d.log" 2>&1 &
  nohup timeout 345 python scripts/g1.py --data "$d" > "/tmp/ex/g1_$d.log" 2>&1 &
done
sleep 2; echo started; date
