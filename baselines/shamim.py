#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import yaml

# Dynamically load system package paths for BCC and libraries in virtualenv
config_path = "config.yaml"
if os.path.exists(config_path):
    try:
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        paths = cfg.get("system_package_paths", ["/usr/lib/python3/dist-packages"])
        for p in paths:
            if p not in sys.path:
                sys.path.append(p)
    except Exception:
        pass
if "/usr/lib/python3/dist-packages" not in sys.path:
    sys.path.append("/usr/lib/python3/dist-packages")

import argparse
import signal
import time
import pickle
import datetime
import numpy as np
import pandas as pd
import ctypes as ct
from bcc import BPF
from sklearn.feature_extraction.text import TfidfTransformer
from scipy.spatial.distance import cdist

# Configurable constants
SYSCALL_TO_IDX = {sid: i for i, sid in enumerate(range(134))}
NUM_FEATURES = len(SYSCALL_TO_IDX)
VECTOR_SIZE = NUM_FEATURES + 1  # 135 features (last one for overflow)
SYSCALL_THRESHOLD = 400

# eBPF Kernel program
BPF_TEXT = f"""
#include <uapi/linux/ptrace.h>
#include <linux/sched.h>

BPF_HASH(excluded_pids, u32, u8, 131072);
BPF_ARRAY(syscall_hist_0, u32, {VECTOR_SIZE});
BPF_ARRAY(syscall_hist_1, u32, {VECTOR_SIZE});
BPF_ARRAY(control_map, u32, 2);

struct event_t {{
    u32 ready_buf;
}};
BPF_PERF_OUTPUT(events);

TRACEPOINT_PROBE(raw_syscalls, sys_enter) {{
    u64 pid_tgid = bpf_get_current_pid_tgid();
    u32 tgid = pid_tgid >> 32;

    u8 *excluded = excluded_pids.lookup(&tgid);
    if (excluded) {{
        return 0;
    }}

    u32 id = args->id;
    if (id >= {VECTOR_SIZE - 1}) {{
        id = {VECTOR_SIZE - 1}; 
    }}

    u32 zero = 0, one = 1;
    u32 *active_idx = control_map.lookup(&zero);
    u32 *total_count = control_map.lookup(&one);
    
    if (!active_idx || !total_count) return 0;
    u32 current_idx = *active_idx;

    if (current_idx == 0) {{
        u32 *val = syscall_hist_0.lookup(&id);
        if (val) lock_xadd(val, 1);
    }} else {{
        u32 *val = syscall_hist_1.lookup(&id);
        if (val) lock_xadd(val, 1);
    }}

    lock_xadd(total_count, 1);
    
    if (*total_count >= {SYSCALL_THRESHOLD}) {{
        *total_count = 0;
        *active_idx = 1 - current_idx;
        struct event_t evt = {{}};
        evt.ready_buf = current_idx;
        events.perf_submit(args, &evt, sizeof(evt));
    }}
    return 0;
}}
"""

# PID Exclusion utilities
def read_proc_stat(pid):
    try:
        path = f"/proc/{pid}/stat"
        with open(path, "r") as f:
            s = f.read().strip()
        rparen = s.rfind(")")
        after = s[rparen + 2:].split()
        return int(after[1])  # PPID
    except:
        return 0

def read_proc_comm(pid):
    try:
        path = f"/proc/{pid}/comm"
        with open(path, "r") as f:
            return f.read().strip()
    except:
        return "?"

def get_ancestor_chain(pid):
    chain = []
    cur = pid
    while cur > 0:
        comm = read_proc_comm(cur)
        chain.append((cur, comm))
        ppid = read_proc_stat(cur)
        if ppid == cur or ppid == 0:
            break
        cur = ppid
    return chain

def find_nearest_bash_ancestor(pid):
    chain = get_ancestor_chain(pid)
    for p, comm in chain:
        if comm == "bash":
            return p
    return None

def get_all_processes():
    proc_info = {}
    try:
        for name in os.listdir("/proc"):
            if not name.isdigit():
                continue
            pid = int(name)
            ppid = read_proc_stat(pid)
            comm = read_proc_comm(pid)
            proc_info[pid] = {"ppid": ppid, "comm": comm}
    except Exception:
        pass
    return proc_info

def build_children_map(proc_info):
    children = {}
    for pid, info in proc_info.items():
        children.setdefault(info["ppid"], []).append(pid)
    return children

def get_descendants(root_pid, children_map):
    result, stack = set(), [root_pid]
    while stack:
        cur = stack.pop()
        if cur in result:
            continue
        result.add(cur)
        for ch in children_map.get(cur, []):
            stack.append(ch)
    return result

def print_subtree(root_pid, proc_info, children_map, prefix="", is_last=True):
    comm = proc_info.get(root_pid, {}).get("comm", "?")
    connector = "└─" if is_last else "├─"
    print(f"{prefix}{connector}{comm}({root_pid})")
    childs = sorted(children_map.get(root_pid, []))
    for i, ch in enumerate(childs):
        print_subtree(ch, proc_info, children_map, prefix + ("  " if is_last else "│ "), i == len(childs) - 1)

def load_excluded_pids(bpf_obj, pid_set):
    excluded = bpf_obj["excluded_pids"]
    one = ct.c_uint8(1)
    for pid in pid_set:
        try:
            excluded[ct.c_uint32(pid)] = one
        except Exception:
            pass

class ShamimPipeline:
    def __init__(self):
        self.running = True
        self.buffer = []

    def handle_signal(self, signum, frame):
        print("\n[*] Interrupted! Stopping...")
        self.running = False

    def collect(self, duration, output_dir, flush_interval):
        signal.signal(signal.SIGINT, self.handle_signal)
        signal.signal(signal.SIGTERM, self.handle_signal)

        os.makedirs(output_dir, exist_ok=True)
        print(f"[*] Starting Shamim collection daemon...")
        print(f"[*] Storing Parquets in: {output_dir}")

        # 1. PID Exclusion tree resolving
        self_pid = os.getpid()
        exclude_root_pid = find_nearest_bash_ancestor(self_pid) or self_pid
        proc_info = get_all_processes()
        children_map = build_children_map(proc_info)
        excluded_pids = get_descendants(exclude_root_pid, children_map)
        
        print(f"[*] Excluding process tree rooted at {read_proc_comm(exclude_root_pid)}({exclude_root_pid}):")
        print_subtree(exclude_root_pid, proc_info, children_map)
        print(f"[*] Excluded PID set contains {len(excluded_pids)} processes.")

        # 2. Compile & load eBPF
        print("[*] Loading eBPF program...")
        b = BPF(text=BPF_TEXT)
        load_excluded_pids(b, excluded_pids)

        last_flush = time.time()
        start_time = time.time()

        def process_event(cpu, data, size):
            event = b["events"].event(data)
            ready_buf = event.ready_buf
            hist_map = b[f"syscall_hist_{ready_buf}"]

            f_vector = np.zeros(VECTOR_SIZE, dtype=np.int32)
            for k, v in hist_map.items():
                if v.value > 0:
                    if k.value < VECTOR_SIZE:
                        f_vector[k.value] = v.value
                    # Reset mapping value in kernel
                    hist_map[k] = ct.c_uint32(0)

            # Record event timestamp and histogram frequency counts
            record = {"time_window": int(time.time() * 1e9)}
            for idx, val in enumerate(f_vector):
                record[f"f_{idx}"] = val
            self.buffer.append(record)

        b["events"].open_perf_buffer(process_event, page_cnt=4096)
        print("[*] Collection active. Press Ctrl+C to terminate.")

        def flush_to_disk():
            if not self.buffer:
                return
            df = pd.DataFrame(self.buffer)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(output_dir, f"shamim_{timestamp}.parquet")
            print(f"[*] Flushing {len(self.buffer)} records to: {filepath}")
            df.to_parquet(filepath, index=False)
            self.buffer.clear()

        try:
            while self.running:
                b.perf_buffer_poll(timeout=100)
                
                # Dynamic Duration boundary exit
                if duration and (time.time() - start_time >= duration):
                    print(f"[*] Collection duration limit ({duration}s) reached.")
                    break
                
                # Check flush interval
                if time.time() - last_flush >= flush_interval:
                    flush_to_disk()
                    last_flush = time.time()
        except KeyboardInterrupt:
            pass
        finally:
            flush_to_disk()
            print("[*] Collection stopped cleanly.")

    def train(self, data_dir, model_dir, percentile):
        os.makedirs(model_dir, exist_ok=True)
        print(f"[*] Starting Shamim training on datasets under: {data_dir}")

        if not os.path.exists(data_dir):
            print(f"[!] Error: Data directory '{data_dir}' does not exist.")
            sys.exit(1)

        parquet_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".parquet")]
        if not parquet_files:
            print(f"[!] Error: No parquet files found in data lake directory: {data_dir}")
            sys.exit(1)

        print(f"[*] Discovered {len(parquet_files)} parquet data files. Loading...")
        dfs = []
        for pf in parquet_files:
            try:
                dfs.append(pd.read_parquet(pf))
            except Exception as e:
                print(f"[!] Warning: Failed to read {pf}: {e}")

        if not dfs:
            print("[!] Error: No valid data could be loaded.")
            sys.exit(1)

        df = pd.concat(dfs, ignore_index=True)
        print(f"[*] Loaded training dataset with shape: {df.shape}")

        # Features columns resolution (f_0 to f_134)
        feature_cols = [f"f_{i}" for i in range(VECTOR_SIZE)]
        missing_cols = set(feature_cols) - set(df.columns)
        if missing_cols:
            print(f"[!] Warning: Missing {len(missing_cols)} feature columns in parquet. Filling with 0.")
            for c in missing_cols:
                df[c] = 0

        X_raw = df[feature_cols].values.astype(np.float32)

        # 1. Log-normalise raw frequency vectors
        print("[*] Log-normalizing raw counts...")
        X_tf = np.log1p(X_raw)

        # 2. Fit TF-IDF transformer
        print("[*] Transforming features via TF-IDF...")
        transformer = TfidfTransformer(sublinear_tf=False)
        X_tfidf = transformer.fit_transform(X_tf).toarray()

        # 3. Compute centroid of training dataset
        print("[*] Computing centroid vector...")
        final_centroid = np.mean(X_tfidf, axis=0).reshape(1, -1)

        # 4. Compute distances of all training instances to the centroid
        print("[*] Calculating distance profile...")
        distances = cdist(final_centroid, X_tfidf, metric='euclidean').flatten()

        # 5. Determine distance percentile threshold
        threshold = np.percentile(distances, percentile)
        print(f"[*] Training finished.")
        print(f"    - Number of points: {len(X_tfidf)}")
        print(f"    - Distance min/max: {np.min(distances):.6f} / {np.max(distances):.6f}")
        print(f"    - Centroid distance threshold ({percentile}th percentile): {threshold:.6f}")

        # Save model objects
        model_path = os.path.join(model_dir, "model.pkl")
        model_state = {
            "transformer": transformer,
            "centroid": final_centroid,
            "threshold": threshold,
            "percentile": percentile
        }
        with open(model_path, "wb") as f:
            pickle.dump(model_state, f)
        print(f"[*] Model checkpoint successfully saved: {model_path}")

    def monitor(self, model_dir, check_interval):
        signal.signal(signal.SIGINT, self.handle_signal)
        signal.signal(signal.SIGTERM, self.handle_signal)

        model_path = os.path.join(model_dir, "model.pkl")
        if not os.path.exists(model_path):
            print(f"[!] Error: Model checkpoint file not found at: {model_path}")
            print("[*] Please run the 'train' command before executing 'monitor'.")
            sys.exit(1)

        print(f"[*] Loading model checkpoint: {model_path}")
        with open(model_path, "rb") as f:
            model_state = pickle.load(f)

        transformer = model_state["transformer"]
        final_centroid = model_state["centroid"]
        threshold = model_state["threshold"]
        percentile = model_state.get("percentile", 99.0)

        print(f"[*] Model parameters successfully loaded:")
        print(f"    - Detection threshold ({percentile}th percentile): {threshold:.6f}")

        # 1. Resolve host exclusions
        self_pid = os.getpid()
        exclude_root_pid = find_nearest_bash_ancestor(self_pid) or self_pid
        proc_info = get_all_processes()
        children_map = build_children_map(proc_info)
        excluded_pids = get_descendants(exclude_root_pid, children_map)
        
        print(f"[*] Excluding process tree rooted at {read_proc_comm(exclude_root_pid)}({exclude_root_pid}):")
        print_subtree(exclude_root_pid, proc_info, children_map)
        print(f"[*] Exclusion set: {len(excluded_pids)} processes.")

        # 2. Compile & load eBPF probe
        print("[*] Starting in-kernel BCC tracepoint compiler...")
        b = BPF(text=BPF_TEXT)
        load_excluded_pids(b, excluded_pids)

        def process_event(cpu, data, size):
            event = b["events"].event(data)
            ready_buf = event.ready_buf
            hist_map = b[f"syscall_hist_{ready_buf}"]

            f_vector = np.zeros(VECTOR_SIZE, dtype=np.float32)
            for k, v in hist_map.items():
                if v.value > 0:
                    if k.value < VECTOR_SIZE:
                        f_vector[k.value] = v.value
                    hist_map[k] = ct.c_uint32(0)

            # Anomaly scoring pipeline
            tf_vector = np.log1p(f_vector).reshape(1, -1)
            tfidf_vector = transformer.transform(tf_vector).toarray()
            distance = cdist(final_centroid, tfidf_vector, metric='euclidean').flatten()[0]

            ts_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            if distance > threshold:
                print(f"[{ts_str}] [🚨 ALARM] Anomaly Detected! Score: {distance:.6f} > Threshold: {threshold:.6f}")
            else:
                print(f"[{ts_str}] [INFO] Normal syscall segment. Score: {distance:.6f}")

        b["events"].open_perf_buffer(process_event, page_cnt=4096)
        print(f"[*] Shamim Real-Time Monitor engine active. Scanning events...")

        try:
            while self.running:
                b.perf_buffer_poll(timeout=int(check_interval * 1000))
        except KeyboardInterrupt:
            pass
        print("[*] Monitor stopped cleanly.")

def main():
    parser = argparse.ArgumentParser(
        description="Shamim et al. Intrusion Detection Baseline Model (eBPF + TF-IDF)",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", required=True, help="Baseline operating modes")

    # Collect Command
    collect_parser = subparsers.add_parser("collect", help="Attach eBPF counts logs and output Parquet datasets")
    collect_parser.add_argument("--duration", type=int, default=0, help="Graceful exit duration in seconds (0 = run until Ctrl+C)")
    collect_parser.add_argument("--output-dir", type=str, default="checkpoint/data_lake/shamim", help="Path to output dataset files")
    collect_parser.add_argument("--interval", type=float, default=1.0, help="Polling timer check intervals (seconds)")
    collect_parser.add_argument("--flush-interval", type=float, default=60.0, help="Flushing buffer file intervals (seconds)")

    # Train Command
    train_parser = subparsers.add_parser("train", help="Incorporate collected parquets into centroids mapping")
    train_parser.add_argument("--data-dir", type=str, default="checkpoint/data_lake/shamim", help="Path to input datasets")
    train_parser.add_argument("--model-dir", type=str, default="checkpoint/shamim", help="Path to store trained serialized checkpoints")
    train_parser.add_argument("--percentile", type=float, default=99.0, help="Distance percentile anomaly thresholds boundary")

    # Monitor Command
    monitor_parser = subparsers.add_parser("monitor", help="Attach live BCC probe and score anomalies against centroids")
    monitor_parser.add_argument("--model-dir", type=str, default="checkpoint/shamim", help="Path to load trained serialization checkpoints")
    monitor_parser.add_argument("--interval", type=float, default=1.0, help="Performance polling checks intervals (seconds)")

    args = parser.parse_args()
    pipeline = ShamimPipeline()

    if args.command == "collect":
        pipeline.collect(duration=args.duration, output_dir=args.output_dir, flush_interval=args.flush_interval)
    elif args.command == "train":
        pipeline.train(data_dir=args.data_dir, model_dir=args.model_dir, percentile=args.percentile)
    elif args.command == "monitor":
        pipeline.monitor(model_dir=args.model_dir, check_interval=args.interval)

if __name__ == "__main__":
    main()
