# Quantum Link-Layer Protocol Simulation Code

This repository contains the simulation code accompanying our paper on **quantum link-layer protocols**, "A Code-Centric Design Space for Error Management over Quantum Network Links". It provides reference circuit implementations for multiple quantum codes, scripts to reproduce the experiments reported in the paper, and a hybrid error management framework that combines filtering and decoding strategies.

> **At a glance**
>
> * **Codes implemented:** Surface code, Steane code, Repetition code
> * **Protocols:** Entanglement distillation (ED), entanglement encoding with transmission (EE-T), entanglement encoding with non-local CNOT gates (EE-NC), state encoding (SE)
> * **Hybrid error management:** Syndrome-weight filter (SW), logical-gap filter of the matching decoder (DS + MWPM), neural-network confidence filter (DS + NN)
> * **Artifacts:** Logical-error calculators, experiment drivers, plotting scripts, pre-trained NN models

---

## Figures of the paper

The repository contains code only. Each script caches its results in CSV files and computes only the points that are missing, so a second run only plots. Sample sizes are those used in the paper (appendix E). The data plotted in the paper are available from the authors upon reasonable request.

**Logical error rate.** For every protocol the reported logical error rate is the average of the Z-basis and X-basis failure probabilities (conditioned on acceptance). The protocol-comparison scripts compute both bases; the hybrid state-encoding circuits are symmetric under the exchange of the two bases (checked at d = 3, 5, 7, 9 with the MWPM-based filters), so the hybrid scripts simulate the Z basis.

| Figure | Content | Script |
|---|---|---|
| 4 | ED, EE-T and SE with error correction | `three-protocols/experiments/error_correction_of_three_protocols/plot_column_chart.py` |
| 5 | EE-T versus EE-NC | `three-protocols/experiments/EE-T_and_2G/compare_correction.py` |
| 6 | Filters with the MWPM and NN decoders, d = 3 | `hybrid-error-management/decoder_comparison_d3.py` |
| 7 | Hybrid error management versus three-layer distillation | `hybrid-error-management/compare_with_distillation.py` |
| 8 | DS + NN over transmission error rate and target acceptance | `hybrid-error-management/near_term_smallcode.py` (sweep), `plot_3d_landscape_nn.py` (plot) |
| 10 | Error detection with three codes | `three-protocols/experiments/error_detection_of_three_codes/plot_column_chart.py` |
| 11 | Finite-size behavior, d = 5 to 11 | `hybrid-error-management/finite_size_scan/` |
| 12 | Runtime of the filter-decoder pairs | `hybrid-error-management/runtime_benchmark/` |

---

## Repository layout

```
.
├── three-protocols/                         # Protocol comparison experiments
│   ├── circuits/
│   │   ├── DEJMPS.py                        # Entanglement distillation (DEJMPS) circuits
│   │   ├── steane.py                        # Steane [[7,1,3]] code circuits
│   │   └── surface.py                       # Surface code circuits
│   ├── cals/
│   │   └── cal_logical_error.py             # Logical error estimation utilities
│   └── experiments/
│       ├── error_correction_of_three_protocols/
│       │   └── plot_column_chart.py         # ED, EE-T and SE under error correction (figure 4)
│       ├── EE-T_and_2G/
│       │   └── compare_correction.py        # EE-T versus EE-NC under error correction (figure 5)
│       └── error_detection_of_three_codes/
│           └── plot_column_chart.py         # Repetition, surface and Steane codes under error detection (figure 10)
│
├── hybrid-error-management/                 # Hybrid filtering + decoding experiments
│   ├── Model/
│   │   ├── model.py                         # Transformer-based NN architecture
│   │   └── model_d_{3,5,7,9,11}.pt          # Pre-trained model weights
│   ├── surface.py                           # Surface code state encoding with parameterized noise
│   ├── estimator_weight.py                  # SW filter + PM decoder
│   ├── estimator_z.py                       # DS filter + PM decoder
│   ├── estimator_NN.py                      # DS filter + NN decoder
│   ├── estimator_swNN.py                    # SW filter + NN decoder
│   ├── near_term_smallcode.py               # Small code (d=3) sweep over local error, transmission error and acceptance
│   ├── decoder_comparison_d3.py             # Filters with the MWPM and NN decoders at d=3 (figure 6)
│   ├── compare_with_distillation.py         # Hybrid filters versus three-layer distillation (figure 7)
│   ├── plot_3d_landscape_nn.py              # 3D plot of DS + NN (figure 8)
│   ├── threshold_scan.py                    # Threshold scan across multiple code distances
│   ├── finite_size_scan/                    # Finite-size scan with at least 100 failures per point (figure 11)
│   ├── runtime_benchmark/                   # Runtime benchmark of the filter-decoder pairs (figure 12)
│   ├── plot_3d_landscape.py                 # 3D landscape visualization (all filters)
│   └── plot_running_time.py                 # Quick runtime estimate
│
└── requirements.txt                         # Python dependencies
```

---

## Installation

**Python version.** Python 3.10+ is recommended.

```bash
# 1) Create a clean environment (recommended)
python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)

# 2) Install dependencies
pip install -r requirements.txt
```

If you use conda:

```bash
conda create -n qlink python=3.10 -y
conda activate qlink
pip install -r requirements.txt
```

**Note on PyTorch.** The `hybrid-error-management` experiments require PyTorch. If you need GPU support, install the appropriate CUDA variant following the [PyTorch installation guide](https://pytorch.org/get-started/locally/) before running `pip install -r requirements.txt`.

---

## Part 1: Three Protocols (`three-protocols/`)

This folder compares quantum link-layer protocols across different codes.

### Circuits (`circuits/`)

* **`surface.py`** -- Surface code circuits, including logical state preparation and stabilizer measurement.
* **`steane.py`** -- Steane [[7,1,3]] code circuits.
* **`DEJMPS.py`** -- Entanglement distillation primitives (DEJMPS protocol).

### Logical error calculators (`cals/`)

* **`cal_logical_error.py`** -- Estimates logical error rates under different protocol and noise settings. Supports PyMatching (MWPM) for the surface code and a lookup-table decoder for the Steane code (for each syndrome, the more probable logical outcome under the circuit's detector error model, i.e. exact maximum-likelihood decoding). Every surface code and Steane code circuit takes `basis='Z'` (input |0>, Z readout, or the Z_L Z_L parity of a Bell pair) or `basis='X'` (input |+>, X readout, or the X_L X_L parity); the final data readout is noiseless in both. In error detection, a run is accepted only if Alice's and Bob's outcomes agree in all syndrome comparisons performed by the protocol; accepted runs are then decoded with the decoder of the error-correction experiments.

### Experiments (`experiments/`)

* **`error_correction_of_three_protocols/`** -- ED, EE-T and SE under error correction (figure 4). Every surface code protocol has one syndrome-measurement round after transmission.
* **`EE-T_and_2G/`** -- EE-T versus EE-NC under error correction (figure 5).
* **`error_detection_of_three_codes/`** -- Repetition, surface and Steane codes under error detection (figure 10). The Steane code circuit compares both X- and Z-type checks.

**Usage.** Navigate to the relevant experiment folder and run the Python script. Results are cached in `results/`, and only missing points are computed; the figures are saved as PDF files in the experiment folder.

---

## Part 2: Hybrid Error Management (`hybrid-error-management/`)

This folder implements and evaluates hybrid error management strategies that combine a **filter** (to discard high-risk shots) with a **decoder** (to correct remaining errors).

### Filter / estimator modules

| Module | Method | Description |
|---|---|---|
| `estimator_weight.py` | SW + PM | Accepts shots with normalized syndrome weight below a threshold. Decoder-agnostic. |
| `estimator_z.py` | DS + PM | Computes the exact coset gap via DEM lifting. Accepts shots with large gap. |
| `estimator_NN.py` | DS + NN | Uses a pre-trained transformer model to predict logical error probability. Can serve as both filter and decoder. |
| `estimator_swNN.py` | SW + NN | Applies syndrome-weight filtering first, then decodes accepted shots with the NN. |

### Neural network model (`Model/`)

* **`model.py`** -- Transformer-based architecture with convolutional embedding, multi-head self-attention, and dilated 2D convolutions. Input shape: `(B, 3, d+1, d+1)`.
* **`model_d_{3,5,7,9,11}.pt`** -- Pre-trained weights for code distances 3 through 11.

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
python threshold_scan.py --p-local 0.003 --d-list 3,5,7,9 --p-trans-grid 0.01,0.02,0.03,...,0.20
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

---

## Reproducibility checklist

1. Install dependencies from `requirements.txt` in a clean environment.
2. Use the code on the `main` branch, which corresponds to the revised paper; the code of the submitted version is in the commit history.
3. For `three-protocols`: navigate to each experiment subfolder and run its script.
4. For `hybrid-error-management`: run each experiment script as described above. Results are cached in CSV files for incremental computation.

---

## Version history

* **v2.0** (revised paper) -- Scripts for every numerical figure of the paper. Logical error rates averaged over the Z and X bases (X-basis versions of all protocol-comparison circuits); exact maximum-likelihood decoding for the Steane code; distillation baseline with noiseless parity readout. Break-even line of figures 6 and 8 at 2 p_trans / 3, the probability that transmission flips the readout of an unencoded qubit prepared in |0> or |+> and read out in the same basis. One syndrome-measurement round after transmission for every surface code protocol. X- and Z-type checks and the acceptance rule of the paper in the error-detection experiment. Array shapes in the unfiltered-MWPM evaluation of `near_term_smallcode.py`. Import in `estimator_NN.py`. SciPy added to the requirements.
* **v1.0** -- Code of the submitted version.

---

## Contact

For questions and reproducibility issues, please open a GitHub issue or contact the authors.
