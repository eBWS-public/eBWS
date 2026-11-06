# -*- coding: utf-8 -*-
import os
import time
import signal
import datetime
import pandas as pd
import ctypes as ct

from ebws.utils import (
    load_config,
    setup_bcc_paths,
    fix_sudo_ownership,
    get_excluded_pids,
    load_excluded_pids_to_bpf,
)


class DataCollector:
    def __init__(self, config_path="config.yaml", **overrides):
        self.config = load_config(config_path)
        for k, v in overrides.items():
            if v is not None:
                self.config[k] = v

        setup_bcc_paths(self.config)

        self.data_lake_dir = self.config.get("data_lake_dir", "checkpoint/data_lake")
        os.makedirs(self.data_lake_dir, exist_ok=True)
        fix_sudo_ownership(self.data_lake_dir)

        default_tw = float(self.config.get("time_window", 1.0))
        self.collector_interval = float(self.config.get("collector_interval", default_tw))
        self.flush_interval = float(self.config.get("collector_flush_interval", 60.0))

        # Load eBPF logger C code
        script_dir = os.path.dirname(os.path.abspath(__file__))
        c_code_path = os.path.join(script_dir, "logger.c")
        with open(c_code_path, "r") as f:
            self.bpf_text = f.read()

        self.running = True
        self.buffer = []
        self.last_flush_time = time.time()

    def handle_signal(self, signum, frame):
        print("\n[*] Stopping Collector...")
        self.running = False

    def flush_to_parquet(self):
        if not self.buffer:
            return

        df = pd.DataFrame(self.buffer)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}.parquet"
        filepath = os.path.join(self.data_lake_dir, filename)

        print(f"[*] Flushing {len(self.buffer)} records to Parquet: {filepath}")
        df.to_parquet(filepath, engine="pyarrow", index=False)
        fix_sudo_ownership(filepath)
        self.buffer.clear()
        self.last_flush_time = time.time()

    def start(self):
        from bcc import BPF

        signal.signal(signal.SIGINT, self.handle_signal)
        signal.signal(signal.SIGTERM, self.handle_signal)

        excluded_pids = get_excluded_pids()
        print(f"[*] Total excluded PIDs (Self & shell ancestor subtree): {len(excluded_pids)}")

        print("[*] Loading eBPF Probe...")
        bpf_obj = BPF(text=self.bpf_text)
        load_excluded_pids_to_bpf(bpf_obj, excluded_pids)

        counts = [bpf_obj["counts_0"], bpf_obj["counts_1"]]
        control_map = bpf_obj["control_map"]
        current_idx = 0

        print(
            f"[*] Collector started (interval={self.collector_interval}s, "
            f"flush={self.flush_interval}s). Data lake: {self.data_lake_dir}"
        )
        print("[*] Monitoring system calls in background. Press Ctrl+C to stop.")

        try:
            while self.running:
                next_idx = 1 - current_idx
                control_map[ct.c_uint32(0)] = ct.c_uint32(next_idx)

                time.sleep(self.collector_interval)
                current_time_window = int(time.time())

                inactive_map = counts[current_idx]
                items = inactive_map.items()

                for k, v in items:
                    count_sum = sum(v)
                    if count_sum > 0:
                        self.buffer.append({
                            "time_window": current_time_window,
                            "pid": k.pid,
                            "start_t": k.start_t,
                            "ppid": k.ppid,
                            "pstart_t": k.pstart_t,
                            "syscall_id": k.syscall_id,
                            "count": count_sum,
                        })

                inactive_map.clear()
                current_idx = next_idx

                if time.time() - self.last_flush_time >= self.flush_interval:
                    self.flush_to_parquet()

        finally:
            self.flush_to_parquet()
            print("[*] Collector stopped cleanly.")


if __name__ == "__main__":
    collector = DataCollector()
    collector.start()
