# Quantum Link Protocol Simulations

This repository provides circuit-level simulations of error management over noisy quantum network links. It compares entanglement distillation, encoded-entanglement distribution, and direct transmission of encoded states. It also implements hybrid error management: filtering transmission attempts using syndrome information and decoding the accepted attempts.

The code was developed by members of the [Research Center for Quantum Software](https://github.com/cqs-thu) at the Department of Computer Science and Technology, Tsinghua University. It includes quantum-code circuits, experiment and plotting scripts, and pretrained neural-network decoders.

This repository accompanies our paper, [A code-centric design space for error management over quantum network links](https://doi.org/10.1088/2058-9565/aebe5a), in *Quantum Science and Technology*. The paper describes the common coding structure of these protocols and evaluates hybrid error management. The simulation data and supporting measurements are included in [data/](data/).

[Getting started](#getting-started) · [Experiments](#experiments-and-figures) · [Repository layout](#repository-layout) · [Contact](#contact)

---

## Getting started

Use Python 3.11 or 3.12. The pinned NumPy version requires Python 3.11 or newer.

```bash
git clone https://github.com/cqs-thu/code-centric-network-links.git
cd code-centric-network-links
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, create the environment with `py -3.11 -m venv .venv` and activate it with `.venv\Scripts\Activate.ps1` in PowerShell.

If you use conda, create the environment with `conda create -n qlink python=3.11`, run `conda activate qlink`, and then install `requirements.txt` from the repository root.

The hybrid experiments use PyTorch. For GPU support, follow the [PyTorch installation guide](https://pytorch.org/get-started/locally/) and select a CUDA build of the version pinned in `requirements.txt`. The finite-size scan and runtime benchmark launchers use Bash and expect a CUDA GPU for their NN workers.

### Run a small example

From the repository root:

```bash
cd hybrid-error-management
python decoder_comparison_d3.py --quick
```

This compares unfiltered decoding with syndrome-weight and decoder-confidence filtering at code distance 3, using 2,000 calibration and evaluation shots per point. It saves CSV results and the plots `decoder_compare_PM.pdf` and `decoder_compare_NN.pdf` under `hybrid-error-management/smoke_test/fig6/`. The NN decoder uses a CUDA GPU when available and otherwise runs on the CPU. Omit `--quick` to run the full experiment.

---

## Experiments and figures

The experiment scripts generate CSV results and PDF plots. The protocol comparisons and the hybrid decoder-comparison, distillation-comparison, and acceptance-sweep scripts reuse cached results and compute missing points when rerun. The finite-size scan and runtime benchmark have separate workflows described below. The [data manifest](data/manifest.json) lists the recorded datasets, parameters, columns, checksums, and cache locations. To reuse a recorded CSV with an experiment script, copy it to the corresponding `cache_path` listed in the manifest.

**Logical error rate.** For every protocol the reported logical error rate is the average of the Z-basis and X-basis failure probabilities (conditioned on acceptance). The protocol-comparison scripts compute both bases. The hybrid scripts simulate the Z basis, supported by X/Z-basis comparisons at d = 3, 5, 7, 9 with MWPM decoding and the SW and DS filters.

| Figure | Content | Script |
|---|---|---|
| 4 | ED, EE-T and SE with error correction | [plot_column_chart.py](three-protocols/experiments/error_correction_of_three_protocols/plot_column_chart.py) |
| 5 | EE-T versus EE-NC | [compare_correction.py](three-protocols/experiments/EE-T_and_2G/compare_correction.py) |
| 6 | Filters with the MWPM and NN decoders, d = 3 | [decoder_comparison_d3.py](hybrid-error-management/decoder_comparison_d3.py) |
| 7 | Hybrid error management versus three-layer distillation | [compare_with_distillation.py](hybrid-error-management/compare_with_distillation.py) |
| 8 | DS + NN over transmission error rate and target acceptance | [near_term_smallcode.py](hybrid-error-management/near_term_smallcode.py) (sweep), [plot_3d_landscape_nn.py](hybrid-error-management/plot_3d_landscape_nn.py) (plot) |
| 10 | Error detection with three codes | [plot_column_chart.py](three-protocols/experiments/error_detection_of_three_codes/plot_column_chart.py) |
| 11 | Finite-size behavior, d = 5 to 11 | [finite_size_scan/](hybrid-error-management/finite_size_scan/) |
| 12 | Runtime of the filter-decoder pairs | [runtime_benchmark/](hybrid-error-management/runtime_benchmark/) |

---

## Repository layout

```text
.
├── three-protocols/
│   ├── circuits/                 # Repetition, Steane and surface code circuits
│   ├── cals/                     # Logical error estimation
│   └── experiments/              # Protocol comparisons
├── hybrid-error-management/
│   ├── Model/                    # Neural decoder architecture and weights
│   ├── finite_size_scan/
│   └── runtime_benchmark/
├── data/                         # Simulation results and supporting measurements
└── requirements.txt
```

---

## Part 1: Three Protocols (`three-protocols/`)

The protocol comparisons evaluate entanglement distillation (ED), entanglement encoding with transmission (EE-T), entanglement encoding with non-local CNOT gates (EE-NC), and state encoding (SE). They use repetition codes, the Steane [[7,1,3]] code, and the rotated surface [[9,1,3]] code.

`cals/cal_logical_error.py` estimates logical error rates under different protocol and noise settings. It uses PyMatching (MWPM) for the surface code and exact maximum-likelihood lookup-table decoding for the Steane code. Every surface code and Steane code circuit takes `basis='Z'` (input |0>, Z readout, or the Z_L Z_L parity of a Bell pair) or `basis='X'` (input |+>, X readout, or the X_L X_L parity); the final data readout is noiseless in both.

Every surface code protocol has one syndrome-measurement round after transmission. In error detection, a run is accepted only if Alice's and Bob's outcomes agree in all syndrome comparisons performed by the protocol; accepted runs are then decoded with the decoder of the error-correction experiments. The Steane code circuit compares both X- and Z-type checks.

**Usage.** Navigate to the relevant experiment folder and run the Python script. Results are cached in `results/`, and only missing points are computed; the figures are saved as PDF files in the experiment folder.

---

## Part 2: Hybrid Error Management (`hybrid-error-management/`)

The filter and decoder combinations are implemented in the following modules. SW uses the syndrome weight, while DS uses the decoder's confidence in its prediction.

| Module | Method | Description |
|---|---|---|
| `estimator_weight.py` | SW + PM | Accepts shots with normalized syndrome weight below a threshold. Decoder-agnostic. |
| `estimator_z.py` | DS + PM | Computes the exact coset gap via DEM lifting. Accepts shots with large gap. |
| `estimator_NN.py` | DS + NN | Uses a pre-trained transformer model to predict logical error probability. Can serve as both filter and decoder. |
| `estimator_swNN.py` | SW + NN | Applies syndrome-weight filtering first, then decodes accepted shots with the NN. |

The neural decoder in `Model/model.py` uses convolutional embedding, multi-head self-attention, and dilated 2D convolutions, with input shape `(B, 3, d+1, d+1)`. Pretrained weights are provided in `Model/model_d_{3,5,7,9,11}.pt`.

### Experiment scripts

Run the commands below from `hybrid-error-management/`.

#### `decoder_comparison_d3.py` -- Figure 6

Unfiltered decoding, SW and DS filters with the MWPM and NN decoders at d = 3, p_local = 0.01 and 50% target acceptance.

```bash
python decoder_comparison_d3.py            # cache in cache_decoder_comparison/, figs_decoder_comparison/decoder_compare_{PM,NN}.pdf
python decoder_comparison_d3.py --quick    # smoke test with 2000 shots per point (smoke_test/fig6)
```

#### `compare_with_distillation.py` -- Figure 7

Hybrid filters versus three-layer distillation at the acceptance probability of the distillation baseline, p_local = 0.005 and 0.01. The distillation baseline is computed with `cal_DEJMPS_detection_per_basis` of `three-protocols/cals/cal_logical_error.py` (output parities read out without noise; average of the ZZ- and XX-parity errors).

```bash
python compare_with_distillation.py          # cache in cache_distill_compare/, figs_distill_compare/distill_vs_hybrid_d3_plocal_{0.005,0.010}.pdf
python compare_with_distillation.py --quick  # smoke test with 2000 shots per point (smoke_test/fig7)
```

#### `near_term_smallcode.py` and `plot_3d_landscape_nn.py` -- Figure 8

`near_term_smallcode.py` computes all filter + decoder combinations at d = 3 over a grid of local error rates, transmission error rates and target acceptance rates (cache in `cache_nearterm/`). `plot_3d_landscape_nn.py` draws the DS + NN landscape.

```bash
python near_term_smallcode.py      # sweep, cache in cache_nearterm/
python plot_3d_landscape_nn.py     # figure 8 from cache_nearterm/: fig_out/Fig3D_NN_surface_lines_d3.pdf
```

#### `finite_size_scan/` -- Figure 11

SW + MWPM, DS + MWPM and DS + NN at d = 5, 7, 9, 11, p_local = 0.003 and target acceptance 10%, 20%, 40% and 80%. For each point, evaluation shots are added in chunks until every target acceptance has at least 100 logical failures (at most 4e8 shots per point). Workers claim work items through files, so several CPU workers and one worker per GPU can run at the same time and can be restarted.

```bash
cd finite_size_scan
./launch.sh 8 "0"                              # 8 CPU workers (SW, DS + MWPM) and one DS + NN worker on GPU 0
python status.py                               # progress
python finalize.py                             # pool the chunks: out/threshold_scan_*_acc*.csv, out/fig11_all_points.csv
python plot_compare_filters_by_acc.py out      # compare_acc{10,20,40,80}.pdf
```

`FIG11_TEST=1 ./launch.sh 2 "0"` runs the same pipeline with small sample sizes.

#### `runtime_benchmark/` -- Figure 12

Per-round processing time of SW + MWPM and DS + MWPM on one CPU core and of DS + NN on a GPU and on one CPU core (one forward pass per round), in batches of 500 rounds, with warm-up and repeated timed batches.

```bash
cd runtime_benchmark
CORES=0,1,2,3 GPU=0 ./run_benchmark.sh                        # benchmark, tables and figs/runtime.pdf
```

Absolute times depend on the hardware; the paper used Intel Xeon Platinum 8358P CPUs and an NVIDIA GeForce RTX 4090 GPU.

#### `threshold_scan.py` -- Threshold scan

Scans over multiple code distances and transmission error rates with a fixed number of shots. It also provides the circuit and estimator construction used by `finite_size_scan/`.

```bash
python threshold_scan.py threshold-scan --p-local 0.003 --d-list 3,5,7,9 --p-trans-grid 0.01,0.02,0.03,0.05,0.07,0.10,0.15,0.20
```

Key CLI arguments:

* `--d-list` -- Code distances to sweep (default: `3,5,7,9`)
* `--p-local` -- Fixed local error rate (required)
* `--p-trans-grid` -- Transmission error rate grid
* `--accept-list` -- Target acceptance rates (default: `0.10,0.20,0.40,0.80`)
* `--nn-device` -- `cuda:0` or `cpu`
* `--shots-cal`, `--shots-eval` -- Calibration and evaluation shot counts

#### `plot_3d_landscape.py` and `plot_running_time.py`

`plot_3d_landscape.py` draws 3D landscapes of all filters from the caches of `near_term_smallcode.py`; `plot_running_time.py` gives a quick runtime estimate. Figures 8 and 12 of the paper are produced by `plot_3d_landscape_nn.py` and `runtime_benchmark/`.
