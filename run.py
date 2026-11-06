#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys

# If invoked via `sudo python3 run.py` while a local `.venv` exists in the repo root,
# re-exec with `.venv/bin/python3` so `sudo`'s secure_path does not drop virtualenv packages.
_LOCAL_VENV_PY = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".venv", "bin", "python3")
if os.path.exists(_LOCAL_VENV_PY) and os.path.realpath(sys.executable) != os.path.realpath(_LOCAL_VENV_PY):
    os.execv(_LOCAL_VENV_PY, [_LOCAL_VENV_PY] + sys.argv)

import json
import argparse
import numpy as np

from ebws.utils import load_config


def print_info(config_path, bws_dir_override=None):
    cfg = load_config(config_path)
    bws_dir = bws_dir_override or cfg.get("bws_dir", "checkpoint/BWS")
    data_lake_dir = cfg.get("data_lake_dir", "checkpoint/data_lake")

    print("=== eBWS Configuration Summary ===")
    print(f"  Config file             : {config_path}")
    print(f"  Data lake directory     : {data_lake_dir}")
    print(f"  BWS checkpoint directory: {bws_dir}")
    print(f"  Time window (Delta t)   : {cfg.get('time_window', 1.0)} s")
    print(f"  Monitor interval        : {cfg.get('monitor_interval', cfg.get('time_window', 1.0))} s")
    print(f"  Collector interval      : {cfg.get('collector_interval', cfg.get('time_window', 1.0))} s")
    print(f"  Collector flush interval: {cfg.get('collector_flush_interval', 60.0)} s")
    print(f"  PL-Tree max size        : {cfg.get('max_tree_size', 50000)}")
    print(f"  Min cluster radius (eps): {cfg.get('min_radius', 1e-6)}")
    print(f"  tau_dist method         : {cfg.get('tau_dist_method', 'kneedle')} (default={cfg.get('default_tau_dist', 0.04)})")
    print(f"  tau_score method        : {cfg.get('tau_score_method', 'tukey')} (Tukey k={cfg.get('tukey_multiplier', 3.0)})")
    print(f"  Aging eviction threshold: {cfg.get('aging_eviction_threshold', 3600)} intervals")

    meta_path = os.path.join(bws_dir, "bws_meta.json")
    centroids_path = os.path.join(bws_dir, "centroids.npy")
    radii_path = os.path.join(bws_dir, "radii.npy")
    diameters_path = os.path.join(bws_dir, "diameters.npy")
    r_path = radii_path if os.path.exists(radii_path) else diameters_path

    if os.path.exists(meta_path) and os.path.exists(centroids_path) and os.path.exists(r_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)
        centroids = np.load(centroids_path)
        radii = np.load(r_path)
        anchors_info = meta.get("anchors_info", [])
        n_manual = sum(1 for a in anchors_info if a.get("is_manual", False))

        print("\n=== Active BWS Model Summary ===")
        print(f"  Vector dimension        : {meta.get('vector_size')} ({len(meta.get('syscall_to_idx', {}))} known syscalls + 1 unknown)")
        print(f"  Trained time window     : {meta.get('time_window', 1.0)} s")
        print(f"  Selected tau_dist       : {meta.get('tau_dist')} (method: {meta.get('tau_dist_method', 'kneedle')})")
        print(f"  Selected tau_score      : {meta.get('tau_score'):.6f} (method: {meta.get('tau_score_method', 'tukey')})")
        print(f"  Total BWS anchors (k)   : {len(centroids)} ({len(centroids) - n_manual} learned, {n_manual} manual FP)")
        if len(radii) > 0:
            print(f"  Anchor radii [min/med/max]: [{np.min(radii):.6f} / {np.median(radii):.6f} / {np.max(radii):.6f}]")
    else:
        print(f"\n[*] No trained BWS model found at '{bws_dir}'. Run `python run.py train` to build one.")


def main():
    parser = argparse.ArgumentParser(
        description="eBWS: Lightweight Host Intrusion Detection System using eBPF and Behavioral Working Set.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", required=True, help="Modes of operation")

    # 1. Collect Mode
    collect_parser = subparsers.add_parser(
        "collect",
        help="Run background worker to collect system call frequencies into data lake.",
    )
    collect_parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    collect_parser.add_argument("--data-lake-dir", type=str, default=None, help="Override data_lake_dir")
    collect_parser.add_argument("--time-window", "--interval", dest="collector_interval", type=float, default=None, help="Polling interval in seconds (default: from config.yaml)")
    collect_parser.add_argument("--flush-interval", dest="collector_flush_interval", type=float, default=None, help="Parquet flush interval in seconds (default: from config.yaml)")

    # 2. Train Mode
    train_parser = subparsers.add_parser(
        "train",
        help="Run offline BWS model builder using data lake parquet files.",
    )
    train_parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    train_parser.add_argument("--data-lake-dir", type=str, default=None, help="Override data_lake_dir")
    train_parser.add_argument("--bws-dir", type=str, default=None, help="Override bws_dir")
    train_parser.add_argument("--parquet", type=str, default=None, help="Train from a specific Parquet file instead of the entire data_lake_dir")
    train_parser.add_argument("--train-ratio", type=float, default=0.8, help="Ratio of time windows used for training vs validation (default: 0.8)")
    train_parser.add_argument("--train-range", type=float, nargs=2, metavar=("START", "END"), default=None, help="Explicit time_window range [START, END) for training")
    train_parser.add_argument("--val-range", type=float, nargs=2, metavar=("START", "END"), default=None, help="Explicit time_window range [START, END) for validation")
    train_parser.add_argument("--time-window", type=float, default=None, help="Histogram aggregation window size in seconds")
    train_parser.add_argument("--min-radius", type=float, default=None, help="Minimum cluster radius epsilon (default: 1e-6)")
    train_parser.add_argument("--retrain", action="store_true", help="Adapt model for concept drift without expanding feature dimensions")
    train_parser.add_argument("--threshold", "--tau-dist", dest="threshold", type=float, default=None, help="Force a specific distance threshold (tau_dist) instead of Kneedle selection")
    train_parser.add_argument("--tau-dist-method", choices=["kneedle", "fixed"], default=None, help="Method for selecting tau_dist")
    train_parser.add_argument("--tau-score", type=float, default=None, help="Force a specific anomaly score threshold (tau_score)")
    train_parser.add_argument("--tau-score-method", choices=["tukey", "kmeans", "fixed"], default=None, help="Method for calibrating tau_score")
    train_parser.add_argument("--tukey-multiplier", type=float, default=None, help="IQR multiplier k for Tukey's fence (default: 3.0)")

    # 3. Monitor Mode
    monitor_parser = subparsers.add_parser(
        "monitor",
        help="Run production real-time BWS anomaly detection engine.",
    )
    monitor_parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    monitor_parser.add_argument("--bws-dir", type=str, default=None, help="Override bws_dir")
    monitor_parser.add_argument("--time-window", "--interval", dest="monitor_interval", type=float, default=None, help="Detection polling interval in seconds")
    monitor_parser.add_argument("--max-tree-size", type=int, default=None, help="Maximum number of nodes in the Process Lineage Tree (PL-Tree)")
    monitor_parser.add_argument("--min-radius", type=float, default=None, help="Minimum cluster radius epsilon")
    monitor_parser.add_argument("--tau-score", type=float, default=None, help="Override anomaly score threshold (tau_score)")
    monitor_parser.add_argument("--aging-threshold", dest="aging_eviction_threshold", type=int, default=None, help="Unmatched intervals before aging eviction")

    # 4. Add False Positive Mode
    add_fp_parser = subparsers.add_parser(
        "add-fp",
        help="Mark a false positive process as a manual benign BWS anchor.",
    )
    add_fp_parser.add_argument("--pid", type=int, required=True, help="PID of the false positive process")
    add_fp_parser.add_argument("--start-t", type=int, default=0, help="Start time (ticks/ns since boot) of the process (default: 0)")
    add_fp_parser.add_argument("--radius", type=float, default=None, help="Custom radius for the manual anchor (default: model tau_dist)")
    add_fp_parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    add_fp_parser.add_argument("--data-lake-dir", type=str, default=None, help="Override data_lake_dir")
    add_fp_parser.add_argument("--bws-dir", type=str, default=None, help="Override bws_dir")

    # 5. Info Mode
    info_parser = subparsers.add_parser(
        "info",
        help="Display current eBWS configuration and trained BWS model summary.",
    )
    info_parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    info_parser.add_argument("--bws-dir", type=str, default=None, help="Override bws_dir")

    args = parser.parse_args()

    if args.command == "collect":
        print("[*] Starting eBWS Data Collection daemon...")
        from ebws.collector import DataCollector

        collector = DataCollector(
            config_path=args.config,
            data_lake_dir=args.data_lake_dir,
            collector_interval=args.collector_interval,
            collector_flush_interval=args.collector_flush_interval,
        )
        collector.start()

    elif args.command == "train":
        print("[*] Starting BWS training pipeline...")
        from ebws.trainer import BWSTrainer

        trainer = BWSTrainer(
            config_path=args.config,
            data_lake_dir=args.data_lake_dir,
            bws_dir=args.bws_dir,
            time_window=args.time_window,
            min_radius=args.min_radius,
            tau_dist_method=args.tau_dist_method,
            tau_score_method=args.tau_score_method,
            tukey_multiplier=args.tukey_multiplier,
        )
        trainer.train_bws(
            train_ratio=args.train_ratio,
            is_retrain=args.retrain,
            force_dist_threshold=args.threshold,
            force_score_threshold=args.tau_score,
            train_range=tuple(args.train_range) if args.train_range else None,
            val_range=tuple(args.val_range) if args.val_range else None,
            parquet_path=args.parquet,
        )

    elif args.command == "monitor":
        print("[*] Starting eBWS Production Monitoring engine...")
        from ebws.monitor import RealTimeMonitor

        monitor = RealTimeMonitor(
            config_path=args.config,
            bws_dir=args.bws_dir,
            monitor_interval=args.monitor_interval,
            max_tree_size=args.max_tree_size,
            min_radius=args.min_radius,
            tau_score=args.tau_score,
            aging_eviction_threshold=args.aging_eviction_threshold,
        )
        monitor.start()

    elif args.command == "add-fp":
        print(f"[*] Registering False Positive process footprint: PID {args.pid}, start_t {args.start_t}...")
        from ebws.trainer import BWSTrainer

        trainer = BWSTrainer(
            config_path=args.config,
            data_lake_dir=args.data_lake_dir,
            bws_dir=args.bws_dir,
        )
        trainer.add_fp_anchor(pid=args.pid, start_t=args.start_t, radius=args.radius)

    elif args.command == "info":
        print_info(config_path=args.config, bws_dir_override=args.bws_dir)


if __name__ == "__main__":
    main()
