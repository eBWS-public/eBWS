# -*- coding: utf-8 -*-
import os
import glob
import json
import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_distances
from sklearn.cluster import KMeans

try:
    from kneed import KneeLocator
except ImportError:
    class KneeLocator:
        """Built-in Kneedle (Satopaa et al., ICDCSW 2011) elbow locator fallback."""

        def __init__(self, x, y, S=1.0, curve="convex", direction="decreasing", online=False):
            self.x = np.asarray(x, dtype=np.float64)
            self.y = np.asarray(y, dtype=np.float64)
            self.S = float(S)
            self.curve = curve
            self.direction = direction
            self.online = online
            self.knee = None
            self.elbow = None
            self._find_knee()

        def _find_knee(self):
            if len(self.x) < 3:
                return
            sort_idx = np.argsort(self.x)
            x = self.x[sort_idx]
            y = self.y[sort_idx]
            x_min, x_max = np.min(x), np.max(x)
            y_min, y_max = np.min(y), np.max(y)
            if x_max <= x_min or y_max <= y_min:
                return
            x_norm = (x - x_min) / (x_max - x_min)
            y_norm = (y - y_min) / (y_max - y_min)
            if self.direction == "decreasing":
                y_diff = (1.0 - y_norm) - x_norm if self.curve == "convex" else y_norm - (1.0 - x_norm)
            else:
                y_diff = y_norm - x_norm if self.curve == "concave" else x_norm - y_norm

            maxima_ids = set(argrelextrema(y_diff, np.greater_equal)[0])
            step = self.S * np.mean(np.diff(x_norm))
            candidate_idx = None
            threshold = 0.0
            for i in range(1, len(y_diff) - 1):
                if i in maxima_ids and (y_diff[i] > y_diff[i - 1] or y_diff[i] > y_diff[i + 1]):
                    if candidate_idx is None or y_diff[i] >= y_diff[candidate_idx]:
                        candidate_idx = i
                        threshold = y_diff[i] - step
                if candidate_idx is not None and i > candidate_idx and y_diff[i] < threshold:
                    self.knee = x[candidate_idx]
                    if not self.online:
                        break
            if self.knee is None and len(y_diff) > 2:
                best_idx = int(np.argmax(y_diff[1:-1])) + 1
                if y_diff[best_idx] > 0:
                    self.knee = x[best_idx]
            self.elbow = self.knee

HAS_KNEED = True

from ebws.utils import load_config


class BWSTrainer:
    def __init__(self, config_path="config.yaml", **overrides):
        self.config = load_config(config_path)
        for k, v in overrides.items():
            if v is not None:
                self.config[k] = v

        self.data_lake_dir = self.config.get("data_lake_dir", "checkpoint/data_lake")
        self.bws_dir = self.config.get("bws_dir", "checkpoint/BWS")
        os.makedirs(self.bws_dir, exist_ok=True)

        # Time window & clustering parameters
        self.time_window = float(self.config.get("time_window", 1.0))
        self.min_radius = float(self.config.get("min_radius", 1e-6))
        self.manual_anchor_radius = self.config.get("manual_anchor_radius", None)

        # Distance threshold (tau_dist) selection parameters (Sec. III-C)
        self.tau_dist_method = str(self.config.get("tau_dist_method", "kneedle")).lower()
        self.default_tau_dist = float(self.config.get("default_tau_dist", 0.04))
        default_candidates = [0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01]
        self.kneedle_candidates = self.config.get(
            "kneedle_candidates",
            self.config.get("dist_threshold_range", default_candidates),
        )
        self.kneedle_S = float(self.config.get("kneedle_S", 1.0))

        # Anomaly score threshold (tau_score) calibration parameters (Sec. III-C)
        self.tau_score_method = str(self.config.get("tau_score_method", "tukey")).lower()
        self.tukey_multiplier = float(self.config.get("tukey_multiplier", 3.0))
        self.default_tau_score = float(self.config.get("default_tau_score", 4.06))

    def load_data_lake(self, parquet_path=None):
        """Loads and combines all Parquet files in the data lake (or a specific file), normalizing schemas."""
        if parquet_path:
            parquet_files = [parquet_path]
        else:
            parquet_files = sorted(glob.glob(os.path.join(self.data_lake_dir, "*.parquet")))
        if not parquet_files:
            raise FileNotFoundError(f"No Parquet files found in data lake: {self.data_lake_dir}")

        print(f"[*] Loading {len(parquet_files)} parquet file(s) from data lake...")
        dfs = []
        for f in parquet_files:
            df_single = pd.read_parquet(f)
            # Schema mapping/normalization for historic vs double-buffered formats
            if "time" in df_single.columns and "time_window" not in df_single.columns:
                df_single = df_single.rename(columns={"time": "time_window"})
            if "start_t" not in df_single.columns:
                df_single["start_t"] = 0
            if "pstart_t" not in df_single.columns:
                df_single["pstart_t"] = 0
            dfs.append(df_single)

        df = pd.concat(dfs, ignore_index=True)
        return df

    def build_histograms(self, df, syscall_to_idx, unknown_idx, vector_size):
        """Pivots raw system call event counts into per-time-window, per-process L2-normalized vectors."""
        if len(df) == 0:
            return np.empty((0, vector_size), dtype=np.float32)

        df_mapped = df.copy()
        df_mapped["mapped_idx"] = (
            df_mapped["syscall_id"].map(syscall_to_idx).fillna(unknown_idx).astype(int)
        )

        # Bucket by configured time_window (Sec. III-B: each process within each time window)
        if self.time_window <= 0 or self.time_window == 1.0:
            df_mapped["time_bucket"] = df_mapped["time_window"]
        else:
            df_mapped["time_bucket"] = np.floor(
                df_mapped["time_window"].astype(np.float64) / self.time_window
            ).astype(np.int64)

        pivot = df_mapped.pivot_table(
            index=["time_bucket", "pid", "start_t"],
            columns="mapped_idx",
            values="count",
            aggfunc="sum",
            fill_value=0,
        )

        expected_cols = list(range(vector_size))
        pivot = pivot.reindex(columns=expected_cols, fill_value=0)

        X = pivot.values.astype(np.float32)
        X_norm = normalize(X, norm="l2")
        return X_norm

    def build_bws_tree(self, X, min_dist_threshold=0.01):
        """Builds a single hierarchical BWS tree down to min_dist_threshold using 2-step FPS (Algorithm 1)."""
        n_points = X.shape[0]
        if n_points == 0:
            return None

        root = {"indices": np.arange(n_points)}
        queue = [root]

        while queue:
            node = queue.pop(0)
            current_indices = node["indices"]
            cluster_pts = X[current_indices]

            # Calculate centroid, radius, and WCSS for this node
            mean_vec = normalize(np.mean(cluster_pts, axis=0).reshape(1, -1))[0]
            dists = cosine_distances(mean_vec.reshape(1, -1), cluster_pts).flatten()

            node["centroid"] = mean_vec
            node["radius"] = float(max(np.max(dists), self.min_radius))
            node["wcss_sum"] = float(np.sum(dists))

            if len(current_indices) < 2:
                node["max_dist"] = 0.0
                node["left"] = None
                node["right"] = None
                continue

            # 2-Step Farthest Point Sampling (FPS) to approximate cluster diameter
            random_idx = 0
            d_from_random = cosine_distances(
                cluster_pts[random_idx : random_idx + 1], cluster_pts
            ).flatten()

            p1_local_idx = int(np.argmax(d_from_random))
            d_from_p1 = cosine_distances(
                cluster_pts[p1_local_idx : p1_local_idx + 1], cluster_pts
            ).flatten()

            p2_local_idx = int(np.argmax(d_from_p1))
            max_dist = float(d_from_p1[p2_local_idx])
            node["max_dist"] = max_dist

            # Split if cluster diameter exceeds min_dist_threshold
            if max_dist > min_dist_threshold:
                d_from_p2 = cosine_distances(
                    cluster_pts[p2_local_idx : p2_local_idx + 1], cluster_pts
                ).flatten()
                left_mask = d_from_p1 <= d_from_p2

                # Guard against degenerate split when points are identical
                if np.all(left_mask) or not np.any(left_mask):
                    node["left"] = None
                    node["right"] = None
                else:
                    left_node = {"indices": current_indices[left_mask]}
                    right_node = {"indices": current_indices[~left_mask]}
                    node["left"] = left_node
                    node["right"] = right_node
                    queue.append(left_node)
                    queue.append(right_node)
            else:
                node["left"] = None
                node["right"] = None

        return root

    def extract_clusters_for_tau(self, tree_root, tau_dist):
        """Prunes the BWS tree at tau_dist to extract centroids, radii, and Mean WCSS."""
        if tree_root is None:
            return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.float32), 0.0

        centroids = []
        radii = []
        total_wcss = 0.0
        total_points = len(tree_root["indices"])

        queue = [tree_root]
        while queue:
            node = queue.pop(0)
            if node["left"] is None or node["max_dist"] <= tau_dist:
                centroids.append(node["centroid"])
                radii.append(max(node["radius"], self.min_radius))
                total_wcss += node["wcss_sum"]
            else:
                queue.append(node["left"])
                queue.append(node["right"])

        mean_wcss = 2.0 * (total_wcss / total_points) if total_points > 0 else 0.0
        return (
            np.array(centroids, dtype=np.float32),
            np.array(radii, dtype=np.float32),
            float(mean_wcss),
        )

    def get_recursive_bws(self, X, dist_threshold):
        """Implements Algorithm 1 (Recursive FPS training phase) for a given distance threshold."""
        if X.shape[0] == 0:
            return np.empty((0, X.shape[1]), dtype=np.float32), np.empty((0,), dtype=np.float32)
        tree_root = self.build_bws_tree(X, min_dist_threshold=dist_threshold)
        centroids, radii, _ = self.extract_clusters_for_tau(tree_root, dist_threshold)
        return centroids, radii

    def select_tau_dist_kneedle(self, X_train, candidates=None):
        """Selects optimal tau_dist via Kneedle elbow detection on a single BWS tree (Sec. III-C)."""
        if candidates is None:
            candidates = self.kneedle_candidates
        candidates = sorted([float(c) for c in candidates], reverse=True)
        min_cand = min(candidates)

        print(f"[*] Building hierarchical BWS tree down to min_dist_threshold = {min_cand}...")
        tree_root = self.build_bws_tree(X_train, min_dist_threshold=min_cand)

        k_list = []
        wcss_list = []
        cluster_cache = {}

        for td in candidates:
            centroids, radii, mean_wcss = self.extract_clusters_for_tau(tree_root, td)
            k = len(centroids)
            k_list.append(k)
            wcss_list.append(mean_wcss)
            cluster_cache[td] = (centroids, radii, mean_wcss)
            print(f"  tau_dist: {td:<6.4f} | Clusters (k): {k:<5d} | Mean WCSS: {mean_wcss:.6f}")

        # Sort by k ascending for Kneedle
        sort_idx = np.argsort(k_list)
        k_arr = np.array(k_list)[sort_idx]
        wcss_arr = np.array(wcss_list)[sort_idx]
        tau_arr = np.array(candidates)[sort_idx]

        selected_tau = self.default_tau_dist
        if HAS_KNEED and len(np.unique(k_arr)) >= 3 and np.max(wcss_arr) > np.min(wcss_arr):
            try:
                kneedle = KneeLocator(
                    k_arr,
                    wcss_arr,
                    curve="convex",
                    direction="decreasing",
                    S=self.kneedle_S,
                )
                if kneedle.knee is not None:
                    knee_idx = int(np.where(k_arr == kneedle.knee)[0][0])
                    selected_tau = float(tau_arr[knee_idx])
                    print(
                        f"[*] Kneedle selected tau_dist = {selected_tau} "
                        f"(k = {kneedle.knee}, Mean WCSS = {wcss_arr[knee_idx]:.6f})"
                    )
                else:
                    print(f"[*] Kneedle did not find a distinct knee; falling back to default_tau_dist = {selected_tau}")
            except Exception as exc:
                print(f"[*] Kneedle fallback ({exc}); using default_tau_dist = {selected_tau}")
        else:
            print(f"[*] Using default_tau_dist = {selected_tau}")

        if selected_tau in cluster_cache:
            centroids, radii, _ = cluster_cache[selected_tau]
        else:
            centroids, radii, _ = self.extract_clusters_for_tau(tree_root, selected_tau)

        return selected_tau, centroids, radii

    @staticmethod
    def score_samples(X, centroids, radii):
        """Computes normalized anomaly score: min_{(c, r) in BWS} d(z, c) / r (Algorithm 1, Line 17)."""
        if len(X) == 0 or len(centroids) == 0:
            return np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.int64)
        dist_matrix = cosine_distances(X, centroids)
        norm_dist_matrix = dist_matrix / radii
        scores = np.min(norm_dist_matrix, axis=1)
        assigned_clusters = np.argmin(norm_dist_matrix, axis=1)
        return scores.astype(np.float32), assigned_clusters.astype(np.int64)

    def calculate_tukey_threshold(self, scores, iqr_multiplier=None):
        """Calibrates tau_score using Tukey's outer fence Q3 + k * IQR on validation scores > 1.0 (Sec. III-C)."""
        if iqr_multiplier is None:
            iqr_multiplier = self.tukey_multiplier

        scores_arr = np.asarray(scores, dtype=np.float64)
        if len(scores_arr) == 0:
            return float(self.default_tau_score)

        tail_scores = scores_arr[scores_arr > 1.0]
        if len(tail_scores) == 0:
            return float(max(np.max(scores_arr), 1.0))
        if len(tail_scores) < 4:
            return float(np.max(tail_scores))

        q1 = np.percentile(tail_scores, 25)
        q3 = np.percentile(tail_scores, 75)
        iqr = q3 - q1
        upper_fence = q3 + iqr_multiplier * iqr

        valid_inliers = tail_scores[tail_scores <= upper_fence]
        if len(valid_inliers) > 0:
            return float(np.max(valid_inliers))
        return float(upper_fence)

    def calculate_kmeans_threshold(self, scores):
        """Legacy 1D K-Means (k=2) threshold calculation on validation outlier scores."""
        X = np.array(scores).reshape(-1, 1)
        if len(np.unique(X)) < 2:
            return float(np.max(X)) if len(X) > 0 else 1.0

        kmeans = KMeans(n_clusters=2, n_init="auto", random_state=42)
        kmeans.fit(X)

        labels = kmeans.labels_
        centers = kmeans.cluster_centers_.flatten()

        min_cluster_idx = np.argmin(centers)
        min_cluster_scores = X[labels == min_cluster_idx]
        return float(np.max(min_cluster_scores))

    def train_bws(
        self,
        train_ratio=0.8,
        force_dist_threshold=None,
        force_score_threshold=None,
        is_retrain=False,
        train_range=None,
        val_range=None,
        parquet_path=None,
    ):
        """Core training orchestration performing Initial Training or Concept Drift Retraining."""
        df = self.load_data_lake(parquet_path=parquet_path)

        # Split time-windows for Train / Validation sets first so vocabulary is strictly train-only
        if train_range is not None and val_range is not None:
            df_train = df[(df["time_window"] >= train_range[0]) & (df["time_window"] < train_range[1])]
            df_val = df[(df["time_window"] >= val_range[0]) & (df["time_window"] < val_range[1])]
        else:
            time_windows = sorted(df["time_window"].unique())
            split_idx = max(1, int(len(time_windows) * train_ratio))
            if split_idx >= len(time_windows) and len(time_windows) > 1:
                split_idx = len(time_windows) - 1
            train_windows = set(time_windows[:split_idx])
            val_windows = set(time_windows[split_idx:]) if len(time_windows) > 1 else train_windows
            df_train = df[df["time_window"].isin(train_windows)]
            df_val = df[df["time_window"].isin(val_windows)]

        print(f"[*] Splitting: {len(df_train)} train events, {len(df_val)} validation events")

        meta_path = os.path.join(self.bws_dir, "bws_meta.json")
        non_manual_centroids = None
        manual_centroids = None
        manual_radii = None
        num_manual_appended = 0

        if is_retrain and os.path.exists(meta_path):
            print("[*] Concept Drift mode active. Loading existing syscall mappings...")
            with open(meta_path, "r") as f:
                existing_meta = json.load(f)
            syscall_to_idx = {int(k): v for k, v in existing_meta["syscall_to_idx"].items()}
            vector_size = existing_meta["vector_size"]
            unknown_idx = existing_meta["unknown_idx"]
            anchors_info = existing_meta.get("anchors_info", [])

            centroids_path = os.path.join(self.bws_dir, "centroids.npy")
            radii_path = os.path.join(self.bws_dir, "radii.npy")
            diameters_path = os.path.join(self.bws_dir, "diameters.npy")
            r_path = radii_path if os.path.exists(radii_path) else diameters_path

            if os.path.exists(centroids_path) and os.path.exists(r_path):
                old_centroids = np.load(centroids_path)
                old_radii = np.load(r_path)

                manual_indices = [
                    info["cluster_id"] for info in anchors_info if info.get("is_manual", False)
                ]
                non_manual_indices = [
                    info["cluster_id"] for info in anchors_info if not info.get("is_manual", False)
                ]

                if non_manual_indices:
                    non_manual_centroids = old_centroids[non_manual_indices]
                if manual_indices:
                    manual_centroids = old_centroids[manual_indices]
                    manual_radii = old_radii[manual_indices]
                    num_manual_appended = len(manual_centroids)
                    print(f"  Loaded {num_manual_appended} manual FP anchors to preserve directly.")
                    print(
                        f"  Loaded {len(non_manual_centroids) if non_manual_centroids is not None else 0} "
                        f"non-manual anchors for union FPS."
                    )
        else:
            print("[*] Initial Training mode active. Building syscall vocabulary from training split...")
            unique_syscalls = sorted(df_train["syscall_id"].unique())
            syscall_to_idx = {int(sid): i for i, sid in enumerate(unique_syscalls)}
            vector_size = len(syscall_to_idx) + 1
            unknown_idx = vector_size - 1
            anchors_info = []

        X_train = self.build_histograms(df_train, syscall_to_idx, unknown_idx, vector_size)
        X_val = self.build_histograms(df_val, syscall_to_idx, unknown_idx, vector_size)
        print(f"[*] Vectorized samples: {X_train.shape[0]} train, {X_val.shape[0]} validation (dim={vector_size})")

        # Retraining Operational Cycle: X_train = Union of new data and previous non-manual BWS anchors
        if is_retrain and non_manual_centroids is not None and len(non_manual_centroids) > 0:
            print(
                f"[*] Rebuilding BWS: merging {len(non_manual_centroids)} prior non-manual anchors into the FPS pool..."
            )
            X_train = np.vstack([X_train, non_manual_centroids])

        # 1. Select tau_dist and build BWS clusters
        if force_dist_threshold is not None:
            best_tau_dist = float(force_dist_threshold)
            print(f"[*] Training with fixed distance threshold tau_dist = {best_tau_dist}")
            best_centroids, best_radii = self.get_recursive_bws(X_train, best_tau_dist)
        elif self.tau_dist_method == "fixed":
            best_tau_dist = float(self.default_tau_dist)
            print(f"[*] Training with configured fixed tau_dist = {best_tau_dist}")
            best_centroids, best_radii = self.get_recursive_bws(X_train, best_tau_dist)
        else:
            best_tau_dist, best_centroids, best_radii = self.select_tau_dist_kneedle(
                X_train, candidates=self.kneedle_candidates
            )

        if best_centroids is None or len(best_centroids) == 0:
            raise ValueError("Training failed to build any clusters. Adjust your thresholds.")

        # 2. Score validation data using normalized distance rule: min_{(c, r)} d(z, c) / r
        val_scores, _ = self.score_samples(X_val, best_centroids, best_radii)

        # 3. Determine anomaly threshold tau_score
        if force_score_threshold is not None:
            tau_score = float(force_score_threshold)
        elif self.tau_score_method == "fixed":
            tau_score = float(self.default_tau_score)
        elif self.tau_score_method == "kmeans":
            outlier_scores = val_scores[val_scores > 1.0]
            tau_score = (
                self.calculate_kmeans_threshold(outlier_scores)
                if len(outlier_scores) > 0
                else self.default_tau_score
            )
        else:
            tau_score = self.calculate_tukey_threshold(
                val_scores, iqr_multiplier=self.tukey_multiplier
            )

        print(f"[*] Determined anomaly threshold tau_score = {tau_score:.6f} (method: {self.tau_score_method})")

        # 4. Append manually validated FP anchors back into BWS, bypassing recursive FPS
        if is_retrain and manual_centroids is not None and len(manual_centroids) > 0:
            print(
                f"[*] Direct integration: appending {num_manual_appended} manual FP anchors directly to rebuilt BWS..."
            )
            best_centroids = np.vstack([best_centroids, manual_centroids])
            best_radii = np.hstack([best_radii, manual_radii])

        # 5. Update BWS anchors metadata list
        new_anchors_info = []
        for i in range(len(best_centroids)):
            is_manual = bool(
                num_manual_appended > 0 and i >= (len(best_centroids) - num_manual_appended)
            )
            new_anchors_info.append({
                "cluster_id": i,
                "age": 0,
                "is_manual": is_manual,
            })

        # 6. Save artifacts (both radii.npy and diameters.npy for compatibility)
        np.save(os.path.join(self.bws_dir, "centroids.npy"), best_centroids)
        np.save(os.path.join(self.bws_dir, "radii.npy"), best_radii)
        np.save(os.path.join(self.bws_dir, "diameters.npy"), best_radii)

        meta_payload = {
            "vector_size": vector_size,
            "time_window": float(self.time_window),
            "min_radius": float(self.min_radius),
            "tau_dist": float(best_tau_dist),
            "tau_dist_method": self.tau_dist_method if force_dist_threshold is None else "fixed",
            "tau_score": float(tau_score),
            "tau_score_method": self.tau_score_method if force_score_threshold is None else "fixed",
            "tukey_multiplier": float(self.tukey_multiplier),
            "syscall_to_idx": {str(k): v for k, v in syscall_to_idx.items()},
            "unknown_idx": unknown_idx,
            "anchors_info": new_anchors_info,
        }

        with open(meta_path, "w") as f:
            json.dump(meta_payload, f, indent=2)

        print(f"[*] Saved BWS model to: {self.bws_dir}")
        print(f"  Centroids: {best_centroids.shape} | Radii: {best_radii.shape}")
        return meta_payload

    def add_fp_anchor(self, pid, start_t, radius=None):
        """Finds a process footprint in the data lake by PID and start_t,
        and adds it as a manual BWS anchor vector directly to checkpoints.
        """
        df = self.load_data_lake()

        match_df = df[(df["pid"] == pid) & (df["start_t"] == start_t)]
        if len(match_df) == 0 and start_t == 0:
            match_df = df[df["pid"] == pid]

        if len(match_df) == 0:
            all_matches = df[df["pid"] == pid]
            if len(all_matches) > 0:
                available_times = all_matches["start_t"].unique()
                raise ValueError(
                    f"Process PID {pid} found in data lake, but start_t {start_t} did not match. "
                    f"Available start times: {available_times}"
                )
            raise ValueError(
                f"Process PID {pid} and start_t {start_t} not found in data lake parquet files."
            )

        print(f"[*] Found process {pid} footprint in data lake ({len(match_df)} syscall records).")

        meta_path = os.path.join(self.bws_dir, "bws_meta.json")
        centroids_path = os.path.join(self.bws_dir, "centroids.npy")
        radii_path = os.path.join(self.bws_dir, "radii.npy")
        diameters_path = os.path.join(self.bws_dir, "diameters.npy")
        r_path = radii_path if os.path.exists(radii_path) else diameters_path

        if not (os.path.exists(meta_path) and os.path.exists(centroids_path) and os.path.exists(r_path)):
            raise FileNotFoundError(
                "Active model checkpoints not found. Please train initial BWS first."
            )

        with open(meta_path, "r") as f:
            meta = json.load(f)

        vector_size = meta["vector_size"]
        unknown_idx = meta["unknown_idx"]
        syscall_to_idx = {int(k): v for k, v in meta["syscall_to_idx"].items()}
        anchors_info = meta["anchors_info"]

        centroids = np.load(centroids_path)
        radii = np.load(r_path)

        vec = np.zeros(vector_size, dtype=np.float32)
        for _, row in match_df.iterrows():
            sid = int(row["syscall_id"])
            col_idx = syscall_to_idx.get(sid, unknown_idx)
            vec[col_idx] += row["count"]

        norm = np.linalg.norm(vec)
        norm_vec = vec / (norm + 1e-9)

        anchor_r = radius
        if anchor_r is None:
            anchor_r = self.manual_anchor_radius
        if anchor_r is None:
            anchor_r = meta.get("tau_dist", self.default_tau_dist)
        anchor_r = max(float(anchor_r), self.min_radius)

        centroids = np.vstack([centroids, norm_vec])
        radii = np.hstack([radii, np.float32(anchor_r)])

        new_id = len(centroids) - 1
        anchors_info.append({
            "cluster_id": new_id,
            "age": 0,
            "is_manual": True,
        })
        meta["anchors_info"] = anchors_info

        np.save(centroids_path, centroids)
        np.save(radii_path, radii)
        np.save(diameters_path, radii)
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        print(
            f"[+] Process {pid} (start_t={start_t}) successfully added as manual benign anchor "
            f"ID {new_id} (radius={anchor_r:.4f})!"
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="eBWS Offline Model Trainer")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--retrain", action="store_true", help="Adapt model for concept drift")
    parser.add_argument("--threshold", type=float, default=None, help="Force distance threshold (tau_dist)")
    parser.add_argument("--tau-score", type=float, default=None, help="Force anomaly score threshold (tau_score)")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio (default: 0.8)")
    args = parser.parse_args()

    trainer = BWSTrainer(config_path=args.config)
    trainer.train_bws(
        train_ratio=args.train_ratio,
        is_retrain=args.retrain,
        force_dist_threshold=args.threshold,
        force_score_threshold=args.tau_score,
    )
