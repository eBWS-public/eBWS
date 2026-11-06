# -*- coding: utf-8 -*-
import sys
import os
import subprocess
import shutil
import json
import unittest
import numpy as np
import pandas as pd

# Add repository root to path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from ebws.pl_tree import PLTree
from ebws.trainer import BWSTrainer
from ebws.monitor import RealTimeMonitor
from ebws.utils import save_config


class TestE2EPipeline(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.test_dir = os.path.join(REPO_ROOT, "checkpoint", "test_e2e_sandbox")
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        os.makedirs(self.test_dir, exist_ok=True)

        self.data_lake = os.path.join(self.test_dir, "data_lake")
        self.bws = os.path.join(self.test_dir, "BWS")
        os.makedirs(self.data_lake, exist_ok=True)
        os.makedirs(self.bws, exist_ok=True)

        self.config_path = os.path.join(self.test_dir, "config.yaml")

        sim_config = {
            "data_lake_dir": self.data_lake,
            "bws_dir": self.bws,
            "time_window": 1.0,
            "max_tree_size": 10,
            "min_radius": 1.0e-6,
            "monitor_interval": 1.0,
            "collector_interval": 1.0,
            "collector_flush_interval": 5.0,
            "tau_dist_method": "kneedle",
            "default_tau_dist": 0.04,
            "kneedle_candidates": [0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01],
            "kneedle_S": 1.0,
            "tau_score_method": "tukey",
            "tukey_multiplier": 3.0,
            "default_tau_score": 2.0,
            "aging_eviction_threshold": 3,
        }
        save_config(sim_config, self.config_path)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_normalized_scoring_and_tukey_calibration(self):
        """Verifies Algorithm 1 Line 17 normalized distance scoring and Sec. III-C Tukey fence."""
        # Construct 2 anchors:
        # Cluster 0 at [1, 0] with tight radius r0 = 0.001
        # Cluster 1 at [0, 1] with wide radius r1 = 0.100
        centroids = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        radii = np.array([0.001, 0.100], dtype=np.float32)

        # Query point slightly closer in raw cosine distance to Cluster 0 (angle 40 deg vs 50 deg):
        # cos(40 deg) = 0.7660 -> d(z, c0) = 0.2340 -> d(z, c0)/r0 = 234.0
        # cos(50 deg) = 0.6428 -> d(z, c1) = 0.3572 -> d(z, c1)/r1 = 3.572
        theta = np.deg2rad(40.0)
        z = np.array([[np.cos(theta), np.sin(theta)]], dtype=np.float32)

        scores, assigned = BWSTrainer.score_samples(z, centroids, radii)
        self.assertEqual(int(assigned[0]), 1)
        self.assertAlmostEqual(float(scores[0]), (1.0 - np.sin(theta)) / 0.100, places=4)

        # Verify Tukey's outer fence (Q3 + 3.0 * IQR on tail scores > 1.0)
        trainer = BWSTrainer(config_path=self.config_path)
        inlier_tail = np.array([1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0])
        extreme_outlier = np.array([25.0])
        sub_unit = np.array([0.1, 0.5, 0.9, 1.0])
        val_scores = np.concatenate([sub_unit, inlier_tail, extreme_outlier])

        tau_score = trainer.calculate_tukey_threshold(val_scores, iqr_multiplier=3.0)
        # Q1(inlier_tail + [25]) = 1.35, Q3 = 1.85, IQR = 0.50 -> fence = 1.85 + 1.5 = 3.35
        # Max tail score <= fence is 2.0 (filtering out the extreme spike 25.0)
        self.assertAlmostEqual(tau_score, 2.0, places=5)

        # Verify Kneedle KneeLocator on the Jetson Thor (k, Mean WCSS) curve (Fig. 5a: k=485, tau_dist=0.03)
        from ebws.trainer import KneeLocator
        k_arr = np.array([251, 261, 274, 300, 330, 360, 405, 485, 614, 898])
        wcss_arr = np.array([0.002415, 0.002278, 0.002244, 0.001594, 0.001425, 0.001312, 0.001178, 0.000832, 0.000673, 0.000291])
        kl = KneeLocator(k_arr, wcss_arr, curve="convex", direction="decreasing", S=1.0)
        self.assertEqual(int(kl.knee), 485)

    def test_complete_end_to_end_flow(self):
        print("\n\n[*] ========================================================")
        print("[*] Starting Formal eBWS End-to-End Pipeline Verification Suite...")
        print("[*] Flow: Collection -> Training -> Detection & PLTree -> FPadding -> Retraining -> FPEvicting")
        print("[*] ========================================================")

        # ----------------------------------------------------
        # 1. INITIAL DATA COLLECTION
        # ----------------------------------------------------
        print("\n[*] Step 1: Simulating data collection into data lake Parquet...")
        records = []
        for t in range(10):
            for pid in [1000, 2000, 3000]:
                ppid = 999
                if pid == 1000:
                    syscalls = [0, 1]
                elif pid == 2000:
                    syscalls = [2, 3]
                else:
                    syscalls = [4, 5]
                for syscall_id in syscalls:
                    count = int(np.random.randint(45, 50))
                    records.append({
                        "time_window": t,
                        "pid": pid,
                        "start_t": pid * 100,
                        "ppid": ppid,
                        "pstart_t": ppid * 100,
                        "syscall_id": syscall_id,
                        "count": count,
                    })

        df = pd.DataFrame(records)
        initial_parquet_path = os.path.join(self.data_lake, "initial_data.parquet")
        df.to_parquet(initial_parquet_path, engine="pyarrow", index=False)
        self.assertTrue(os.path.exists(initial_parquet_path))
        print(f"  [+] Initial data collected successfully ({len(df)} records across 10 time windows).")

        # ----------------------------------------------------
        # 2. TRAINING INITIAL BWS MODEL
        # ----------------------------------------------------
        print("\n[*] Step 2: Executing initial model training (run.py train)...")
        cmd_train = [
            sys.executable,
            os.path.join(REPO_ROOT, "run.py"),
            "train",
            "--config",
            self.config_path,
            "--train-ratio",
            "0.8",
        ]
        res = subprocess.run(cmd_train, cwd=REPO_ROOT, capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, f"Training failed: {res.stderr}\n{res.stdout}")

        centroids_path = os.path.join(self.bws, "centroids.npy")
        radii_path = os.path.join(self.bws, "radii.npy")
        diameters_path = os.path.join(self.bws, "diameters.npy")
        meta_path = os.path.join(self.bws, "bws_meta.json")

        self.assertTrue(os.path.exists(centroids_path))
        self.assertTrue(os.path.exists(radii_path))
        self.assertTrue(os.path.exists(diameters_path))
        centroids_0 = np.load(centroids_path)
        print(f"  [+] BWS initialized with set size (centroids): {len(centroids_0)}")
        self.assertEqual(len(centroids_0), 3)  # 3 distinct benign process clusters

        with open(meta_path, "r") as f:
            meta_0 = json.load(f)
        tau_score_0 = meta_0["tau_score"]
        print(f"  [+] Initial thresholds determined: tau_dist = {meta_0['tau_dist']}, tau_score = {tau_score_0:.4f}")

        # ----------------------------------------------------
        # 3. DETECTION & MONITORING WITH PLTREE MANAGEMENT
        # ----------------------------------------------------
        print("\n[*] Step 3: Simulating live monitoring & active PLTree lineage tracking...")
        monitor = RealTimeMonitor(config_path=self.config_path)

        # Inject anomalous system call footprint for process PID 6002
        print("  [*] Injecting anomalous process 6002 (syscalls 50 to 52) triggered by parent bash 6001...")
        X_anom = np.zeros((1, monitor.vector_size), dtype=np.float32)
        for syscall_id in [50, 51, 52]:
            col_idx = monitor.syscall_to_idx.get(syscall_id, monitor.unknown_idx)
            X_anom[0, col_idx] += 100

        # Pre-populate PLTree manually to simulate parent mapping
        monitor.pl_tree.record(6000, 600000, 999, 99900, "curl", -1)
        monitor.pl_tree.record(6001, 600100, 6000, 600000, "bash", -1)
        monitor.pl_tree.record(6002, 600200, 6001, 600100, "malware", -1)

        # Score process 6002 using normalized distance rule Score(z) = min_{(c, r)} d(z, c) / r
        X_norm = X_anom / (np.linalg.norm(X_anom) + 1e-9)
        scores, assigned_clusters = monitor.score_vectors(X_norm)
        anomaly_score = float(scores[0])

        print(f"  [+] Process 6002 anomaly score: {anomaly_score:.2f} (tau: {tau_score_0:.2f})")
        self.assertGreater(anomaly_score, tau_score_0)

        # Trigger PLTree lineage reconstruction
        print("  [*] Reconstructing Process Lineage Tree ancestry for PID 6002:")
        monitor.recursive_add_ancestor(6002, 600200, 6001, 600100, 1000.0, is_root_event=True)
        monitor.print_anomaly_tree(6002, 600200)

        chain = monitor.pl_tree.reconstruct_chain(6002, 600200)
        cmds = [n.comm for n in chain]
        self.assertListEqual(cmds[::-1], ["curl", "bash", "malware"])
        stats = monitor.pl_tree.get_stats()
        self.assertEqual(stats["current_size"], 3)
        self.assertEqual(stats["locked_ancestors"], 2)
        print("  [+] PLTree Ancestry lineage resolved and verified successfully!")

        # Simulating telemetry capture: write process 6002 counts to data lake Parquet
        captured_fp = []
        for syscall_id in [50, 51, 52]:
            captured_fp.append({
                "time_window": 11,
                "pid": 6002,
                "start_t": 600200,
                "ppid": 6001,
                "pstart_t": 600100,
                "syscall_id": syscall_id,
                "count": 100,
            })
        pd.DataFrame(captured_fp).to_parquet(
            os.path.join(self.data_lake, "anom_telemetry.parquet"), engine="pyarrow", index=False
        )

        # ----------------------------------------------------
        # 4. FALSE POSITIVE PADDING (add-fp)
        # ----------------------------------------------------
        print("\n[*] Step 4: Marking flagged process 6002 as a manual false positive (run.py add-fp)...")
        cmd_fp = [
            sys.executable,
            os.path.join(REPO_ROOT, "run.py"),
            "add-fp",
            "--config",
            self.config_path,
            "--pid",
            "6002",
            "--start-t",
            "600200",
        ]
        res_fp = subprocess.run(cmd_fp, cwd=REPO_ROOT, capture_output=True, text=True)
        self.assertEqual(res_fp.returncode, 0, f"FP addition failed: {res_fp.stderr}")

        centroids_1 = np.load(centroids_path)
        self.assertEqual(len(centroids_1), 4)

        with open(meta_path, "r") as f:
            meta_1 = json.load(f)
        self.assertTrue(meta_1["anchors_info"][-1]["is_manual"])
        print(f"  [+] BWS set size expanded successfully to: {len(centroids_1)}")
        print("  [+] Flagged process 6002 directly integrated as a manual benign BWS anchor.")

        # ----------------------------------------------------
        # 5. CONCEPT DRIFT RETRAINING
        # ----------------------------------------------------
        print("\n[*] Step 5: Generating concept drift and executing retraining (run.py train --retrain)...")
        drift_records = []
        for t in range(10, 15):
            for pid in [1000, 2000, 3000]:
                ppid = 999
                if pid == 1000:
                    syscalls = [0, 1, 6]
                elif pid == 2000:
                    syscalls = [2, 3, 6]
                else:
                    syscalls = [4, 5, 6]
                for syscall_id in syscalls:
                    count = int(np.random.randint(20, 50))
                    drift_records.append({
                        "time_window": t,
                        "pid": pid,
                        "start_t": pid * 100,
                        "ppid": ppid,
                        "pstart_t": ppid * 100,
                        "syscall_id": syscall_id,
                        "count": count,
                    })
        pd.DataFrame(drift_records).to_parquet(
            os.path.join(self.data_lake, "drift_data.parquet"), engine="pyarrow", index=False
        )

        cmd_retrain = [
            sys.executable,
            os.path.join(REPO_ROOT, "run.py"),
            "train",
            "--config",
            self.config_path,
            "--train-ratio",
            "0.8",
            "--retrain",
        ]
        res_retrain = subprocess.run(cmd_retrain, cwd=REPO_ROOT, capture_output=True, text=True)
        self.assertEqual(res_retrain.returncode, 0, f"Retraining failed: {res_retrain.stderr}")

        centroids_2 = np.load(centroids_path)
        with open(meta_path, "r") as f:
            meta_2 = json.load(f)

        self.assertEqual(meta_2["vector_size"], 7)  # original 6 syscalls + 1 unknown
        self.assertEqual(meta_2["unknown_idx"], 6)
        self.assertEqual(centroids_2.shape[1], 7)
        self.assertTrue(meta_2["anchors_info"][-1]["is_manual"])

        diff = np.abs(centroids_2[-1] - centroids_1[-1])
        self.assertLess(diff.max(), 1e-6)
        print(f"  [+] BWS set size after rebuild: {len(centroids_2)}")
        print("  [+] Manual FP anchor bypassed recursive FPS and was directly integrated with 0.0 error.")

        # ----------------------------------------------------
        # 6. EVICTION LIFECYCLES & INFO COMMAND
        # ----------------------------------------------------
        print("\n[*] Step 6: Testing aging eviction lifecycle and manual anchor locked-preservation...")
        monitor_final = RealTimeMonitor(config_path=self.config_path)
        initial_cnt = len(monitor_final.centroids)

        for cycle in range(4):
            hit_anchors = {0}
            monitor_final.anchors_info[0]["age"] = 0

            evicted_indices = []
            for idx, info in enumerate(monitor_final.anchors_info):
                if idx not in hit_anchors:
                    info["age"] += 1
                    if info["age"] >= monitor_final.aging_threshold and not info.get("is_manual", False):
                        evicted_indices.append(idx)

            if evicted_indices:
                print(f"    [!] Cycle {cycle}: Evicting {len(evicted_indices)} stale non-manual anchors...")
                keep_mask = np.ones(len(monitor_final.centroids), dtype=bool)
                keep_mask[evicted_indices] = False
                monitor_final.centroids = monitor_final.centroids[keep_mask]
                monitor_final.diameters = monitor_final.diameters[keep_mask]
                new_anchors_info = []
                for new_idx, old_idx in enumerate(np.where(keep_mask)[0]):
                    info = monitor_final.anchors_info[old_idx]
                    info["cluster_id"] = new_idx
                    new_anchors_info.append(info)
                monitor_final.anchors_info = new_anchors_info

        final_cnt = len(monitor_final.centroids)
        print(f"  [+] Anchors before aging: {initial_cnt} | Anchors after aging: {final_cnt}")
        self.assertLess(final_cnt, initial_cnt)
        self.assertTrue(monitor_final.anchors_info[-1]["is_manual"])

        # Test `run.py info` CLI
        cmd_info = [
            sys.executable,
            os.path.join(REPO_ROOT, "run.py"),
            "info",
            "--config",
            self.config_path,
        ]
        res_info = subprocess.run(cmd_info, cwd=REPO_ROOT, capture_output=True, text=True)
        self.assertEqual(res_info.returncode, 0, f"Info command failed: {res_info.stderr}")
        self.assertIn("Active BWS Model Summary", res_info.stdout)

        print("\n[*] ========================================================")
        print("[+] SUCCESS: ALL STAGES OF THE END-TO-END PIPELINE VERIFIED!")
        print("[*] ========================================================\n")


if __name__ == "__main__":
    unittest.main()
