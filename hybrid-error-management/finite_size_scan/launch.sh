#!/bin/bash
# Launch (or restart) the workers of the finite-size scan: one GPU worker per listed GPU (DS+NN) and
# N CPU workers (SW+MWPM, DS+MWPM). Safe to re-run: finished chunks are kept, stale claims are removed first.
# Usage: ./launch.sh [n_cpu_workers=8] [gpus="0"]
NCPU=${1:-8}
GPUS=${2:-"0"}
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
if pgrep -u "$USER" -f "worker.py --kind" > /dev/null; then echo "workers already running"; exit 1; fi
$PY janitor.py
mkdir -p logs
for g in $GPUS; do
  CUDA_VISIBLE_DEVICES=$g nohup $PY worker.py --kind gpu --wid g$g >> logs/gpu_$g.log 2>&1 &
done
for i in $(seq 1 $NCPU); do
  CUDA_VISIBLE_DEVICES="" nohup $PY worker.py --kind cpu --wid c$i >> logs/cpu_$i.log 2>&1 &
done
sleep 2
echo "launched $(pgrep -u "$USER" -f "worker.py --kind" | wc -l) workers at $(date)"
