# -*- coding: utf-8 -*-
import os
import time
import json
import signal
import numpy as np
import ctypes as ct
from scipy.spatial.distance import cdist

from ebws.pl_tree import PLTree
from ebws.utils import (
    load_config,
    setup_bcc_paths,
    fix_sudo_ownership,
    get_excluded_pids,
    load_excluded_pids_to_bpf,
    get_cmdline,
    read_proc_stat,
)


class RealTimeMonitor:
    def __init__(self, config_path="config.yaml", manual_anchor_file=None, **overrides):
        self.config = load_config(config_path)
        for k, v in overrides.items():
            if v is not None:
                self.config[k] = v

        setup_bcc_paths(self.config)

        self.bws_dir = self.config.get("bws_dir", "checkpoint/BWS")
        self.max_tree_size = self.config.get("max_tree_size", 50000)
        default_tw = float(self.config.get("time_window", 1.0))
        self.monitor_interval = float(self.config.get("monitor_interval", default_tw))
        self.aging_threshold = int(self.config.get("aging_eviction_threshold", 3600))
        self.min_radius = float(self.config.get("min_radius", 1e-6))
        self.manual_anchor_radius = self.config.get("manual_anchor_radius", None)

        # Load eBWS model parameters
        self.load_model()

        # Override tau_score if explicitly provided via CLI/overrides
        if "tau_score" in overrides and overrides["tau_score"] is not None:
            self.tau_score = float(overrides["tau_score"])

        # Initialize Process Lineage Tree
        self.pl_tree = PLTree(max_size=self.max_tree_size)

        # Load eBPF C code
        script_dir = os.path.dirname(os.path.abspath(__file__))
        c_code_path = os.path.join(script_dir, "logger.c")
        with open(c_code_path, "r") as f:
            self.bpf_text = f.read()

        self.running = True

    @property
    def radii(self):
        return self.diameters

    @radii.setter
    def radii(self, value):
        self.diameters = value

    def load_model(self):
        """Loads BWS anchors and mapping metadata from checkpoints folder."""
        meta_path = os.path.join(self.bws_dir, "bws_meta.json")
        centroids_path = os.path.join(self.bws_dir, "centroids.npy")
        radii_path = os.path.join(self.bws_dir, "radii.npy")
        diameters_path = os.path.join(self.bws_dir, "diameters.npy")
        r_path = radii_path if os.path.exists(radii_path) else diameters_path

        if not (os.path.exists(meta_path) and os.path.exists(centroids_path) and os.path.exists(r_path)):
            raise FileNotFoundError(
                f"Model files not found in BWS directory: {self.bws_dir}. Please run offline training first."
            )

        with open(meta_path, "r") as f:
            self.meta = json.load(f)

        self.vector_size = self.meta["vector_size"]
        self.tau_dist = float(self.meta["tau_dist"])
        self.tau_score = float(self.meta["tau_score"])
        self.syscall_to_idx = {int(k): v for k, v in self.meta["syscall_to_idx"].items()}
        self.unknown_idx = self.meta["unknown_idx"]
        self.anchors_info = self.meta["anchors_info"]

        self.centroids = np.load(centroids_path).astype(np.float32)
        self.diameters = np.maximum(np.load(r_path).astype(np.float32), self.min_radius)

        print("[*] Model loaded successfully.")
        print(f"  Centroids: {self.centroids.shape} | Radii: {self.diameters.shape}")
        print(f"  tau_dist: {self.tau_dist} | tau_score: {self.tau_score:.4f}")

    def save_model_changes(self):
        """Saves current memory model back to checkpoint folder."""
        meta_path = os.path.join(self.bws_dir, "bws_meta.json")
        centroids_path = os.path.join(self.bws_dir, "centroids.npy")
        radii_path = os.path.join(self.bws_dir, "radii.npy")
        diameters_path = os.path.join(self.bws_dir, "diameters.npy")

        np.save(centroids_path, self.centroids)
        np.save(radii_path, self.diameters)
        np.save(diameters_path, self.diameters)

        self.meta["anchors_info"] = self.anchors_info
        with open(meta_path, "w") as f:
            json.dump(self.meta, f, indent=2)
        for p in (self.bws_dir, centroids_path, radii_path, diameters_path, meta_path):
            fix_sudo_ownership(p)

    def add_manual_anchor(self, vector, radius=None, diameter=None):
        """Appends a new manually validated anchor vector to mitigate future FPs."""
        print("[*] Adding manually approved anchor to BWS to mitigate false positive...")

        norm = np.linalg.norm(vector)
        norm_vector = vector / (norm + 1e-9)

        anchor_r = radius if radius is not None else diameter
        if anchor_r is None:
            anchor_r = self.manual_anchor_radius
        if anchor_r is None:
            anchor_r = self.tau_dist
        anchor_r = max(float(anchor_r), self.min_radius)

        self.centroids = np.vstack([self.centroids, norm_vector.reshape(1, -1).astype(np.float32)])
        self.diameters = np.hstack([self.diameters, np.float32(anchor_r)])

        new_id = len(self.centroids) - 1
        self.anchors_info.append({
            "cluster_id": new_id,
            "age": 0,
            "is_manual": True,
        })

        self.save_model_changes()
        print(f"  Added manual anchor ID {new_id} (radius={anchor_r:.4f}). Total anchors now: {len(self.centroids)}.")

    def score_vectors(self, X_norm):
        """Computes normalized anomaly scores per Algorithm 1 (Line 17):
        Score(z) = min_{(c, r) in BWS} d(z, c) / r
        Returns (anomaly_scores, assigned_clusters).
        """
        dist_matrix_eucl = cdist(X_norm, self.centroids, metric="euclidean")
        dist_matrix = 0.5 * (dist_matrix_eucl ** 2)
        safe_radii = np.maximum(self.diameters, self.min_radius)
        norm_dist_matrix = dist_matrix / safe_radii
        anomaly_scores = np.min(norm_dist_matrix, axis=1)
        assigned_clusters = np.argmin(norm_dist_matrix, axis=1)
        return anomaly_scores, assigned_clusters

    def handle_signal(self, signum, frame):
        print("\n[*] Stopping Monitor...")
        self.running = False

    def recursive_add_ancestor(self, pid, start_t, ppid, pstart_t, current_time, is_root_event=True):
        """Recursively traverses parent execution chains to build complete lineage tree nodes."""
        node = self.pl_tree.get_node(pid, start_t)

        if node:
            if is_root_event:
                self.pl_tree.record(pid, start_t, ppid, pstart_t, node.comm, current_time)
            return

        comm = get_cmdline(pid)
        time_val = current_time if is_root_event else -1
        self.pl_tree.record(pid, start_t, ppid, pstart_t, comm, time_val)

        if ppid > 0 and ppid != 1:
            gppid = -1
            gpstart_t = 0

            # If parent node is already tracked in PL-Tree with a valid grandparent, reuse it
            existing_parent = self.pl_tree.get_node(ppid, pstart_t)
            if existing_parent is not None and existing_parent.ppid > 0:
                gppid = existing_parent.ppid
                gpstart_t = existing_parent.pstart_t
            else:
                try:
                    gppid, _ = read_proc_stat(ppid)
                    if gppid > 0:
                        # Prefer existing grandparent node's exact start_t if present in PL-Tree
                        found_gp = False
                        for (k_pid, k_st), _ in self.pl_tree.nodes.items():
                            if k_pid == gppid:
                                gpstart_t = k_st
                                found_gp = True
                                break
                        if not found_gp:
                            _, gpstart_t = read_proc_stat(gppid)
                except Exception:
                    pass

            self.recursive_add_ancestor(
                ppid, pstart_t, gppid, gpstart_t, current_time, is_root_event=False
            )

    def print_anomaly_tree(self, d_pid, d_start_t):
        """Draws a process lineage hierarchy for the flagged process."""
        chain = self.pl_tree.reconstruct_chain(d_pid, d_start_t)
        if not chain:
            return

        chain = chain[::-1]

        print("\n[!] === Detected Anomaly Process Lineage ===")
        indent = ""
        for i, node in enumerate(chain):
            is_last = i == len(chain) - 1
            prefix = "└─ " if is_last else "├─ "
            time_str = f" [Time: {node.time:.2f}]" if node.time != -1 else ""
            print(f"{indent}{prefix}{node.pid} ({node.comm}){time_str}")
            indent += "    " if is_last else "│   "
        print("============================================\n")

    def start(self):
        from bcc import BPF

        signal.signal(signal.SIGINT, self.handle_signal)
        signal.signal(signal.SIGTERM, self.handle_signal)

        excluded_pids = get_excluded_pids()
        print(f"[*] Total excluded PIDs (Self & shell ancestor subtree): {len(excluded_pids)}")

        print("[*] Hooking into Linux kernel system call tracepoint...")
        bpf_obj = BPF(text=self.bpf_text)
        load_excluded_pids_to_bpf(bpf_obj, excluded_pids)

        counts = [bpf_obj["counts_0"], bpf_obj["counts_1"]]
        control_map = bpf_obj["control_map"]
        current_idx = 0

        print(
            f"[*] eBWS Real-Time Detection Loop Started "
            f"(interval={self.monitor_interval}s, max_tree_size={self.pl_tree.max_size}). Watching for anomalies..."
        )

        try:
            while self.running:
                next_idx = 1 - current_idx
                control_map[ct.c_uint32(0)] = ct.c_uint32(next_idx)

                time.sleep(self.monitor_interval)
                current_time = time.time()

                inactive_map = counts[current_idx]
                items = inactive_map.items()

                process_data = {}
                for k, v in items:
                    count_sum = sum(v)
                    if count_sum > 0:
                        proc_key = (k.pid, k.start_t, k.ppid, k.pstart_t)
                        process_data.setdefault(proc_key, {})[k.syscall_id] = count_sum

                if process_data:
                    active_keys = list(process_data.keys())
                    X = np.zeros((len(active_keys), self.vector_size), dtype=np.float32)

                    for row_idx, key in enumerate(active_keys):
                        counts_dict = process_data[key]
                        for sid, val in counts_dict.items():
                            col_idx = self.syscall_to_idx.get(sid, self.unknown_idx)
                            X[row_idx, col_idx] += val

                    norms = np.linalg.norm(X, axis=1, keepdims=True)
                    X_norm = X / (norms + 1e-9)

                    # Normalized distance scoring per Algorithm 1 (Line 17)
                    anomaly_scores, assigned_clusters = self.score_vectors(X_norm)

                    hit_anchors = set()

                    for row_idx, key in enumerate(active_keys):
                        pid, start_t, ppid, pstart_t = key

                        self.recursive_add_ancestor(
                            pid, start_t, ppid, pstart_t, current_time, is_root_event=True
                        )

                        assigned_cluster = int(assigned_clusters[row_idx])
                        anomaly_score = float(anomaly_scores[row_idx])

                        if anomaly_score > self.tau_score:
                            print(
                                f"[!] ANOMALY ALERT: Process {pid} (start_t={start_t}, cmd='{get_cmdline(pid)}') "
                                f"triggered boundary check! Anomaly Score: {anomaly_score:.2f} (tau: {self.tau_score:.2f})"
                            )
                            self.print_anomaly_tree(pid, start_t)
                        else:
                            hit_anchors.add(assigned_cluster)
                            self.anchors_info[assigned_cluster]["age"] = 0

                    # Aging-Based Eviction
                    model_changed = False
                    evicted_indices = []

                    for idx, info in enumerate(self.anchors_info):
                        if idx not in hit_anchors:
                            info["age"] += 1
                            if info["age"] >= self.aging_threshold and not info.get("is_manual", False):
                                evicted_indices.append(idx)

                    if evicted_indices:
                        print(
                            f"[*] Aging Eviction: Removing {len(evicted_indices)} stale anchors due to inactivity..."
                        )
                        keep_mask = np.ones(len(self.centroids), dtype=bool)
                        keep_mask[evicted_indices] = False

                        self.centroids = self.centroids[keep_mask]
                        self.diameters = self.diameters[keep_mask]

                        new_anchors_info = []
                        for new_idx, old_idx in enumerate(np.where(keep_mask)[0]):
                            info = self.anchors_info[old_idx]
                            info["cluster_id"] = new_idx
                            new_anchors_info.append(info)
                        self.anchors_info = new_anchors_info

                        model_changed = True

                    if model_changed:
                        self.save_model_changes()

                inactive_map.clear()
                current_idx = next_idx

        finally:
            print("[*] Monitor stopped cleanly.")


if __name__ == "__main__":
    monitor = RealTimeMonitor()
    monitor.start()
