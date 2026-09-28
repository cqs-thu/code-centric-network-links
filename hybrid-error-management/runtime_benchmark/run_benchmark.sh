#!/bin/bash
# Example run of the runtime benchmark (figure 12) on one machine, one step after the other.
#   CORES: candidate logical CPUs; each timed section runs on one physical core whose two hardware threads are idle
#   GPU:   CUDA device index for the DS+NN GPU run and the DS+NN thresholds
# Usage: CORES=0,1,2,3 GPU=0 ./run_benchmark.sh
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
CORES=${CORES:-0,1,2,3}
GPU=${GPU:-0}
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p logs
# 1) DS+NN thresholds at 20% target acceptance
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU $PY calibrate_nn.py > logs/calibrate_nn.log 2>&1
# 2) SW+MWPM and DS+MWPM on one CPU core
CUDA_VISIBLE_DEVICES="" $PY bench_cpu_mwpm.py --tag B --reps 20 --lat-blocks 10 --cores $CORES > logs/cpu_mwpm_B.log 2>&1
# 3) DS+NN on one CPU core and on the GPU
CUDA_VISIBLE_DEVICES="" $PY bench_nn.py --device cpu --tag B --reps 10 --lat-blocks 5 --cores $CORES > logs/nn_cpu1_B.log 2>&1
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU $PY bench_nn.py --device cuda --tag B --reps 20 --lat-blocks 10 --cores $CORES > logs/nn_gpu_B.log 2>&1
# 4) tables (runtime_results_regen.csv, ...) and figures (figs/runtime.pdf, figs/runtime_latency.pdf)
$PY summarize.py
$PY plot_runtime.py --tag B
