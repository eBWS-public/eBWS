# eBWS: Lightweight Host Intrusion Detection for IoT Edge Devices

**eBWS** is a lightweight host intrusion detection system (HIDS) designed for resource-constrained IoT edge platforms. It aggregates per-process system call frequencies directly in kernel space using double-buffered eBPF per-CPU hash maps, eliminating high-frequency ring-buffer transfers and context-switching overhead. In user space, eBWS transforms windowed syscall counts into $L_2$-normalized behavioral vectors and evaluates them against a compact **Behavioral Working Set (BWS)** of benign anchor clusters. When an anomaly is detected, a bounded **Process Lineage Tree (PL-Tree)** reconstructs the complete parent-child execution chain even after short-lived processes have terminated and vanished from `/proc`.

---

## Key Features & Architecture

```text
eBWS/
├── config.yaml               # Global configuration (time window, PL-Tree size, thresholds, paths)
├── run.py                    # Unified CLI entry point (collect, train, monitor, add-fp, info)
├── requirements.txt          # Python dependencies
├── ebws/
│   ├── logger.c              # eBPF program (raw_tracepoint:sys_enter) with double-buffered maps
│   ├── collector.py          # Kernel-to-Parquet telemetry collection daemon
│   ├── trainer.py            # Offline BWS builder (2-step FPS, Kneedle tau_dist, Tukey tau_score)
│   ├── monitor.py            # Real-time anomaly detector, PL-Tree integration, & aging eviction
│   ├── pl_tree.py            # Bounded LRU Process Lineage Tree with active-ancestor locking
│   └── utils.py              # /proc inspection, PID exclusion, and configuration utilities
├── baselines/
│   ├── shamim.py             # Shamim et al. single-centroid TF-IDF baseline
│   └── velox.py              # VELOX provenance-edge embedding & MLP baseline
├── tests/
│   └── test_e2e_pipeline.py  # End-to-end verification test suite
└── checkpoint/               # Auto-created at runtime (data_lake/ and BWS/ checkpoints)
```

> **Note on `checkpoint/`**: Pre-trained model files and raw telemetry (`checkpoint/data_lake/` and `checkpoint/BWS/`) are intentionally excluded from the repository because system call baselines and process behaviors are specific to each target host, kernel, and IoT workload. Both directories are created automatically when you run `collect` and `train`.

### Core Modules
- **In-Kernel Aggregation (`ebws/logger.c`)**: Attaches to `raw_tracepoint/sys_enter`, keys processes by `(pid, start_time)` to prevent PID-reuse collisions, and alternates between two `BPF_MAP_TYPE_PERCPU_HASH` maps (`counts_0`, `counts_1`) every time window $\Delta t$.
- **Behavioral Working Set Trainer (`ebws/trainer.py`)**:
  - Pivots per-window process syscall counts into $L_2$-normalized vectors in $\mathbb{R}^{|\mathcal{S}|+1}$ (reserving one dimension for unseen syscalls).
  - Partitions training vectors recursively via 2-step Farthest Point Sampling (FPS) and automatically selects the splitting threshold $\tau_{\text{dist}}$ using the **Kneedle** algorithm on a single hierarchical BWS tree.
  - Calibrates the decision threshold $\tau_{\text{score}}$ on validation tail scores ($\text{Score}(z) > 1.0$) using **Tukey's outer fence** ($Q_3 + 3.0 \times \text{IQR}$).
  - Supports **concept-drift retraining** over the union of new benign data and existing non-manual BWS anchors while preserving manually approved false-positive (FP) anchors.
- **Real-Time Monitor & Lifecycle Manager (`ebws/monitor.py`)**:
  - Computes normalized cosine distance scores:
    $$\text{Score}(z) = \min_{(c, r) \in \text{BWS}} \frac{d_{\cos}(z, c)}{r}$$
    where $r = \max\!\big(\max_{x \in P} d_{\cos}(c, x),\, \epsilon\big)$.
  - Evicts stale non-manual anchors after `aging_eviction_threshold` consecutive unmatched windows.
- **Process Lineage Tree (`ebws/pl_tree.py`)**: Maintains an `OrderedDict`-backed LRU cache of process nodes up to `max_tree_size`. Upstream ancestor nodes discovered via `/proc` traversal (`time = -1`) are locked against eviction as long as any descendant remains active in the tree.

---

## Installation

### 1. System Prerequisites (Linux)
Live eBPF telemetry collection and monitoring require a Linux kernel with eBPF support and the **BCC (BPF Compiler Collection)** toolkit:

```bash
# Debian / Ubuntu / Raspberry Pi OS / Jetson Linux
sudo apt-get update
sudo apt-get install -y bpfcc-tools python3-bpfcc linux-headers-$(uname -r)
```

### 2. Python Environment
Install the Python dependencies inside a virtual environment or Conda environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> **Note on `python3-bpfcc` and `sudo` in Virtual/Conda Environments**:
> 1. Because BCC Python bindings are installed as system packages (`/usr/lib/python3/dist-packages`), `eBWS` automatically appends `system_package_paths` from `config.yaml` to `sys.path`.
> 2. On Linux, `sudo` resets `PATH` (`secure_path`). If you use a local `.venv` directory as shown above, `run.py` automatically detects `.venv/bin/python3`. If you use a **Conda** environment or an external virtualenv, pass `$(which python3)` to `sudo`:
>    ```bash
>    sudo $(which python3) run.py collect
>    ```

---

## Configuration (`config.yaml`)

All operational and algorithmic parameters can be configured in `config.yaml` (and overridden via CLI flags in `run.py`):

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `data_lake_dir` | `"checkpoint/data_lake"` | Directory where collected Parquet telemetry files are stored |
| `bws_dir` | `"checkpoint/BWS"` | Directory where trained BWS checkpoints (`centroids.npy`, `radii.npy`, `bws_meta.json`) are saved |
| `time_window` | `1.0` | Aggregation time window $\Delta t$ (seconds) for histogram vectorization |
| `monitor_interval` | `1.0` | Real-time detection polling interval (seconds) |
| `collector_interval` | `1.0` | eBPF double-buffer swap interval (seconds) |
| `collector_flush_interval` | `60.0` | Interval (seconds) between flushing buffered telemetry to Parquet |
| `max_tree_size` | `50000` | Maximum nodes $N_{\max}$ in the PL-Tree (`<= 0` or `null` for unbounded) |
| `min_radius` | `1.0e-6` | Minimum cluster radius $\epsilon$ to prevent division by zero |
| `manual_anchor_radius` | `null` | Radius for manual FP anchors (`null` defaults to the model's `tau_dist`) |
| `tau_dist_method` | `"kneedle"` | Method for selecting $\tau_{\text{dist}}$: `"kneedle"` or `"fixed"` |
| `default_tau_dist` | `0.04` | Fixed or fallback distance threshold $\tau_{\text{dist}}$ |
| `kneedle_candidates` | `[0.10, ..., 0.01]` | Candidate $\tau_{\text{dist}}$ values evaluated on the BWS tree |
| `kneedle_S` | `1.0` | Sensitivity parameter $S$ for Kneedle elbow detection |
| `tau_score_method` | `"tukey"` | Method for calibrating $\tau_{\text{score}}$: `"tukey"`, `"kmeans"`, or `"fixed"` |
| `tukey_multiplier` | `3.0` | IQR multiplier $k$ for Tukey's fence ($Q_3 + k \cdot \text{IQR}$) |
| `default_tau_score` | `4.06` | Fixed or fallback anomaly score threshold $\tau_{\text{score}}$ |
| `aging_eviction_threshold` | `3600` | Unmatched monitoring intervals before evicting stale non-manual anchors |

---

## Quick Start & Usage

### 1. Collect Benign Telemetry
Run the background collector while your normal workload is running (requires root privileges to attach the eBPF `raw_tracepoint/sys_enter` probe). Press `Ctrl+C` when finished to flush buffered records to `checkpoint/data_lake/*.parquet`:

```bash
# Using a local .venv (or replace `python3` with `$(which python3)` for Conda)
sudo $(which python3) run.py collect --flush-interval 10.0
```

Optional overrides:
```bash
sudo $(which python3) run.py collect --time-window 1.0 --flush-interval 60.0 --data-lake-dir checkpoint/data_lake
```

### 2. Train the BWS Model Offline
Build the BWS anchors and automatically calibrate $\tau_{\text{dist}}$ (via Kneedle) and $\tau_{\text{score}}$ (via Tukey's outer fence) from collected Parquet files in `checkpoint/data_lake/`:

```bash
python3 run.py train --train-ratio 0.8
```

Common training options:
```bash
# Train from a specific Parquet file with explicit [start, end) time window splits
python3 run.py train --parquet path/to/trace.parquet --train-range 0 26400 --val-range 26400 27600

# Override time window, minimum radius, or force specific thresholds
python3 run.py train --time-window 1.0 --min-radius 1e-6 --tau-dist 0.03 --tukey-multiplier 3.0
```

### 3. Inspect Configuration & Trained Model
View the active configuration and summary statistics of the trained BWS model (`checkpoint/BWS/`):

```bash
python3 run.py info
```

### 4. Run Real-Time Anomaly Detection
Start the real-time detection engine and PL-Tree lineage tracker (automatically loads the trained BWS anchors and calibrated $\tau_{\text{score}}$ from `checkpoint/BWS/bws_meta.json`):

```bash
sudo $(which python3) run.py monitor
```

Optional runtime overrides (e.g., pass a lower `--tau-score 0.5` to test anomaly alerts and PL-Tree lineage output):
```bash
sudo $(which python3) run.py monitor --time-window 1.0 --max-tree-size 50000 --tau-score 0.5 --aging-threshold 3600
```

### 5. Register a False Positive (FP) Anchor
If a benign process recorded in `checkpoint/data_lake/` triggers a false alarm, register its footprint as a locked manual anchor (`--start-t` is optional if the PID is unique in the data lake):

```bash
python3 run.py add-fp --pid <PID>
# Or with an explicit process start_t:
python3 run.py add-fp --pid <PID> --start-t <START_TIME>
```

### 6. Concept-Drift Retraining
Update the BWS model with newly collected telemetry while preserving the feature dimensionality and all manual FP anchors:

```bash
python3 run.py train --train-ratio 0.8 --retrain
```

---

## Comparative Baselines

Reference implementations of the comparative baselines evaluated in the paper are provided in `baselines/`:

### Shamim et al. (`baselines/shamim.py`)
Single-centroid detector over log-normalized TF-IDF system call histograms:
```bash
# 1. Collect histogram segments
sudo python3 baselines/shamim.py collect --duration 60

# 2. Train centroid and percentile threshold
python3 baselines/shamim.py train --data-dir checkpoint/data_lake/shamim --percentile 95

# 3. Run real-time monitoring
sudo python3 baselines/shamim.py monitor
```

### VELOX (`baselines/velox.py`)
Provenance-edge predictor using Word2Vec node embeddings and a lightweight MLP classifier (requires `torch` and `gensim`):
```bash
# 1. Collect provenance edges
sudo python3 baselines/velox.py collect --duration 60

# 2. Train Word2Vec embeddings and MLP edge predictor
python3 baselines/velox.py train

# 3. Run real-time inference
sudo python3 baselines/velox.py monitor
```

---

## Testing & Verification

Run the end-to-end verification suite (validates normalized distance scoring, Tukey threshold calibration, Kneedle selection, per-window histogram vectorization, PL-Tree ancestry reconstruction, manual FP padding, concept-drift retraining, and aging eviction):

```bash
python3 -m unittest tests/test_e2e_pipeline.py
```
