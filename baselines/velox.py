#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import yaml

# Support loading system package paths (BCC and libraries in virtualenv)
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
import socket
import struct
import uuid
import time
import pickle
import json
import datetime
import numpy as np
import pandas as pd
import ctypes as ct

import torch
import torch.nn as nn
import torch.nn.functional as F
from gensim.models import Word2Vec
from torch_geometric.data import Data

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model parameters
NODE_OUT_DIM = 64
ENCODER_DROPOUT = 0.3
DECODER_DROPOUT = 0.5
SRC_DST_PROJ_COEF = 2
W2V_DIM = 128

# Mappings & dicts
id_to_event = {
    56: "EVENT_OPEN",
    63: "EVENT_READ", 67: "EVENT_READ",
    64: "EVENT_WRITE", 68: "EVENT_WRITE", 66: "EVENT_WRITE",
    221: "EVENT_EXECUTE",
    203: "EVENT_CONNECT",
    207: "EVENT_RECVFROM",
    212: "EVENT_RECVMSG", 243: "EVENT_RECVMSG",
    206: "EVENT_SENDTO",
    211: "EVENT_SENDMSG", 269: "EVENT_SENDMSG"
}

event_list = list(sorted(list(set(id_to_event.values()))))
event_to_idx = {name: i for i, name in enumerate(event_list)}
NUM_EDGE_TYPES = len(event_list)

fd_syscall_map_dict = {
    56: -3, 63: 0, 67: 0, 64: 0, 68: 0, 66: 0,
    221: -2, 203: 0, 207: 0, 212: 0, 243: 0, 206: 0, 211: 0, 269: 0
}

# Namespace constants
NS_SUBJECT = uuid.UUID('6ba7b810-9dad-11d1-80b4-00c04fd430c8')
NS_OBJECT = uuid.UUID('6ba7b811-9dad-11d1-80b4-00c04fd430c8')

# eBPF C program (same as prototype)
BPF_TEXT = """
#include <uapi/linux/ptrace.h>
#include <linux/sched.h>
#include <linux/fs.h>
#include <linux/fdtable.h>
#include <linux/net.h>
#include <net/sock.h>

#define MAX_STRING_SIZE 256

BPF_HASH(syscall_fd_map, u32, s32);
BPF_HASH(active_syscalls, u64, s32);
BPF_HASH(excluded_pids, u32, u8, 1024);

struct event_t {
    u64 ts;
    u32 pid;
    u32 syscall_id;
    char comm[TASK_COMM_LEN];
    u8 is_socket;
    s32 fd;
    char filename[MAX_STRING_SIZE];
    u32 saddr;
    u32 daddr;
    u16 sport;
    u16 dport;
    u16 family;
};
BPF_PERF_OUTPUT(events);

TRACEPOINT_PROBE(raw_syscalls, sys_enter) {
    u64 id = bpf_get_current_pid_tgid();
    u32 tgid = id >> 32;
    if (excluded_pids.lookup(&tgid)) return 0;

    u32 syscall_id = args->id;
    s32 *fd_idx_ptr = syscall_fd_map.lookup(&syscall_id);
    if (!fd_idx_ptr) return 0;
    
    s32 fd_idx = *fd_idx_ptr;

    if (fd_idx == 0) {
        s32 fd = (s32)args->args[0];
        active_syscalls.update(&id, &fd);
    } 
    else if (fd_idx == -2) {
        struct event_t event = {};
        event.ts = bpf_ktime_get_ns();
        event.pid = tgid;
        event.syscall_id = syscall_id;
        bpf_get_current_comm(&event.comm, sizeof(event.comm));
        bpf_probe_read_user_str(&event.filename, sizeof(event.filename), (const char *)args->args[0]);
        events.perf_submit(args, &event, sizeof(event));
    }
    else if (fd_idx == -3) {
        struct event_t event = {};
        event.ts = bpf_ktime_get_ns();
        event.pid = tgid;
        event.syscall_id = syscall_id;
        bpf_get_current_comm(&event.comm, sizeof(event.comm));
        bpf_probe_read_user_str(&event.filename, sizeof(event.filename), (const char *)args->args[1]);
        events.perf_submit(args, &event, sizeof(event));
    }
    return 0;
}

TRACEPOINT_PROBE(raw_syscalls, sys_exit) {
    u64 id = bpf_get_current_pid_tgid();
    s32 *fd_ptr = active_syscalls.lookup(&id);
    if (!fd_ptr) return 0;
    
    s32 fd = *fd_ptr;
    struct event_t event = {};
    event.ts = bpf_ktime_get_ns();
    event.pid = id >> 32;
    event.syscall_id = args->id;
    bpf_get_current_comm(&event.comm, sizeof(event.comm));
    event.fd = fd;

    struct task_struct *task = (struct task_struct *)bpf_get_current_task();
    struct files_struct *files;
    struct fdtable *fdt;
    struct file **fd_array;
    struct file *f;

    bpf_probe_read_kernel(&files, sizeof(files), &task->files);
    bpf_probe_read_kernel(&fdt, sizeof(fdt), &files->fdt);
    bpf_probe_read_kernel(&fd_array, sizeof(fd_array), &fdt->fd);
    bpf_probe_read_kernel(&f, sizeof(f), &fd_array[fd]);

    if (!f) goto cleanup;

    struct inode *f_inode;
    bpf_probe_read_kernel(&f_inode, sizeof(f_inode), &f->f_inode);
    umode_t i_mode;
    bpf_probe_read_kernel(&i_mode, sizeof(i_mode), &f_inode->i_mode);

    if (S_ISSOCK(i_mode)) {
        event.is_socket = 1;
        struct socket *sock;
        struct sock *sk;
        bpf_probe_read_kernel(&sock, sizeof(sock), &f->private_data);
        bpf_probe_read_kernel(&sk, sizeof(sk), &sock->sk);
        if (sk) {
            bpf_probe_read_kernel(&event.family, sizeof(event.family), &sk->__sk_common.skc_family);
            if (event.family == AF_INET) {
                bpf_probe_read_kernel(&event.saddr, sizeof(event.saddr), &sk->__sk_common.skc_rcv_saddr);
                bpf_probe_read_kernel(&event.daddr, sizeof(event.daddr), &sk->__sk_common.skc_daddr);
                bpf_probe_read_kernel(&event.sport, sizeof(event.sport), &sk->__sk_common.skc_num);
                bpf_probe_read_kernel(&event.dport, sizeof(event.dport), &sk->__sk_common.skc_dport);
            }
        }
    } else {
        struct dentry *dentry;
        struct qstr d_name;
        const unsigned char *name_ptr;
        bpf_probe_read_kernel(&dentry, sizeof(dentry), &f->f_path.dentry);
        bpf_probe_read_kernel(&d_name, sizeof(d_name), &dentry->d_name);
        bpf_probe_read_kernel(&name_ptr, sizeof(name_ptr), &d_name.name);
        bpf_probe_read_kernel_str(&event.filename, sizeof(event.filename), name_ptr);
    }
    events.perf_submit(args, &event, sizeof(event));

cleanup:
    active_syscalls.delete(&id);
    return 0;
}
"""

# PID Exclusion resolving helpers
def get_proc_info(pid):
    try:
        with open(f"/proc/{pid}/stat", "r") as f:
            stat = f.read().split()
            ppid = int(stat[3])
        with open(f"/proc/{pid}/comm", "r") as f:
            comm = f.read().strip()
        return ppid, comm
    except:
        return None, None

def find_nearest_bash_ancestor(pid):
    curr_pid = pid
    while curr_pid > 1:
        ppid, comm = get_proc_info(curr_pid)
        if comm == "bash":
            return curr_pid
        if ppid is None:
            break
        curr_pid = ppid
    return None

def get_all_descendants(root_pid):
    descendants = {root_pid}
    try:
        proc_list = [int(d) for d in os.listdir('/proc') if d.isdigit()]
        children_map = {}
        for p in proc_list:
            ppid, _ = get_proc_info(p)
            if ppid:
                children_map.setdefault(ppid, []).append(p)
        stack = [root_pid]
        while stack:
            curr = stack.pop()
            for child in children_map.get(curr, []):
                if child not in descendants:
                    descendants.add(child)
                    stack.append(child)
    except Exception:
        pass
    return descendants

# Process metadata scraper
def get_node_metadata(pid, comm):
    metadata = {
        "type": "process",
        "path": "",
        "cmd_line": comm,
        "remote_ip": "",
        "remote_port": ""
    }
    try:
        exe_path = os.readlink(f"/proc/{pid}/exe")
        metadata["path"] = exe_path
    except Exception:
        metadata["path"] = f"/usr/bin/{comm}"  # reasonable guess for exe path
    
    try:
        with open(f"/proc/{pid}/cmdline", "r") as f:
            cmd = f.read().replace('\x00', ' ').strip()
            if cmd:
                metadata["cmd_line"] = cmd
    except Exception:
        pass
    return metadata

# PyTorch MLP Model Structure
class LinearEncoder(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        return self.dropout(self.lin1(x))

class CustomEdgeMLP(nn.Module):
    def __init__(self, in_dim, out_dim, num_edge_types, proj_coef=2, dropout=0.5):
        super().__init__()
        self.lin_src = nn.Linear(in_dim, in_dim * proj_coef)
        self.lin_dst = nn.Linear(in_dim, in_dim * proj_coef)
        cat_dim = in_dim * proj_coef * 2
        hidden_dim = int(cat_dim * 0.5)
        self.mlp = nn.Sequential(
            nn.Linear(cat_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_edge_types)
        )
    def forward(self, h_src, h_dst):
        h = torch.cat([self.lin_src(h_src), self.lin_dst(h_dst)], dim=-1)
        return self.mlp(h)

class VeloxModel(nn.Module):
    def __init__(self, in_dim, node_out_dim, num_edge_types):
        super().__init__()
        self.encoder = LinearEncoder(in_dim, node_out_dim, dropout=ENCODER_DROPOUT)
        self.decoder = CustomEdgeMLP(node_out_dim, node_out_dim, num_edge_types, 
                                     proj_coef=SRC_DST_PROJ_COEF, dropout=DECODER_DROPOUT)
        
    def forward(self, *args, **kwargs):
        # Determine signature based on input type
        if len(args) == 1 and hasattr(args[0], 'x'):
            g = args[0]
            h = self.encoder(g.x)
            h_src = h[g.edge_index[0]]
            h_dst = h[g.edge_index[1]]
            logits = self.decoder(h_src, h_dst)
            edge_type_classes = g.edge_attr.argmax(dim=1)
            loss = F.cross_entropy(logits, edge_type_classes, reduction='none')
            return loss
        else:
            x = args[0] if len(args) > 0 else kwargs['x']
            edge_index = args[1] if len(args) > 1 else kwargs['edge_index']
            edge_attr_idx = args[2] if len(args) > 2 else kwargs['edge_attr_idx']
            h = self.encoder(x)
            h_src = h[edge_index[0]]
            h_dst = h[edge_index[1]]
            logits = self.decoder(h_src, h_dst)
            loss = F.cross_entropy(logits, edge_attr_idx, reduction='none')
            return loss

class VeloxPipeline:
    def __init__(self):
        self.running = True
        self.edges_buffer = []
        self.vertices_dict = {}

    def handle_signal(self, signum, frame):
        print("\n[*] Interrupted! Stopping...")
        self.running = False

    def collect(self, duration, output_dir, flush_interval):
        # Local imports for BCC execution environment compatibility
        from bcc import BPF

        signal.signal(signal.SIGINT, self.handle_signal)
        signal.signal(signal.SIGTERM, self.handle_signal)

        os.makedirs(output_dir, exist_ok=True)
        print(f"[*] Starting VELOX collection daemon...")
        print(f"[*] Storing CSVs in: {output_dir}")

        # 1. Resolve exclusions
        self_pid = os.getpid()
        bash_ancestor = find_nearest_bash_ancestor(self_pid)
        exclude_set = get_all_descendants(bash_ancestor) if bash_ancestor else {self_pid}

        print(f"[*] Excluding process tree rooted at bash({bash_ancestor}) with {len(exclude_set)} PIDs.")

        # 2. Compile eBPF Probe
        print("[*] Compiling BCC tracepoints...")
        b = BPF(text=BPF_TEXT)

        # Set up active FD mapping configuration
        syscall_fd_map = b.get_table("syscall_fd_map")
        for sc_id, f_idx in fd_syscall_map_dict.items():
            syscall_fd_map[ct.c_uint32(sc_id)] = ct.c_int32(f_idx)

        # Load exclusions in kernel map
        ex_map = b.get_table("excluded_pids")
        for p in exclude_set:
            ex_map[ct.c_uint32(p)] = ct.c_uint8(1)

        last_flush = time.time()
        start_time = time.time()

        def handle_event(cpu, data, size):
            event = b["events"].event(data)
            
            comm_str = event.comm.decode('utf-8', 'replace').strip()
            s_raw = f"{comm_str}:{event.pid}"
            s_uuid_str = str(uuid.uuid5(NS_SUBJECT, s_raw))

            v_name = id_to_event.get(event.syscall_id, "EVENT_UNKNOWN")
            if v_name == "EVENT_UNKNOWN":
                return # Skip untracked edge events

            # Resolve subject metadata
            if s_uuid_str not in self.vertices_dict:
                self.vertices_dict[s_uuid_str] = get_node_metadata(event.pid, comm_str)

            # Resolve object metadata
            if event.is_socket and event.family == socket.AF_INET:
                ip_str = socket.inet_ntoa(struct.pack('<I', event.daddr))
                port_val = socket.ntohs(event.dport)
                o_raw = f"{ip_str}:{port_val}"
                o_uuid_str = str(uuid.uuid5(NS_OBJECT, o_raw))
                
                if o_uuid_str not in self.vertices_dict:
                    self.vertices_dict[o_uuid_str] = {
                        "type": "socket",
                        "path": "",
                        "cmd_line": "",
                        "remote_ip": ip_str,
                        "remote_port": str(port_val)
                    }
            else:
                fn_str = event.filename.decode('utf-8', 'replace').strip() or "unknown"
                o_uuid_str = str(uuid.uuid5(NS_OBJECT, fn_str))
                
                if o_uuid_str not in self.vertices_dict:
                    self.vertices_dict[o_uuid_str] = {
                        "type": "file",
                        "path": fn_str,
                        "cmd_line": "",
                        "remote_ip": "",
                        "remote_port": ""
                    }

            # Record Edge Event
            self.edges_buffer.append({
                "edge_id": str(uuid.uuid4()),
                "src": s_uuid_str,
                "dst": o_uuid_str,
                "event": v_name,
                "time": int(event.ts)
            })

        def handle_lost(lost):
            print(f"\n[!] WARNING: Lost {lost} eBPF events! Buffer capacity overflow.", file=sys.stderr)

        b["events"].open_perf_buffer(handle_event, page_cnt=16384, lost_cb=handle_lost)
        print("[*] Collection active. Press Ctrl+C to stop.")

        def flush_to_disk():
            if not self.edges_buffer:
                return

            v_file = os.path.join(output_dir, "vertex.csv")
            e_file = os.path.join(output_dir, "edge.csv")

            # Load or create vertices
            if os.path.exists(v_file):
                try:
                    existing_v = pd.read_csv(v_file)
                except Exception:
                    existing_v = pd.DataFrame(columns=["node_id", "type", "path", "cmd_line", "remote_ip", "remote_port"])
            else:
                existing_v = pd.DataFrame(columns=["node_id", "type", "path", "cmd_line", "remote_ip", "remote_port"])

            new_v_list = []
            for uid, meta in self.vertices_dict.items():
                new_v_list.append({
                    "node_id": uid,
                    "type": meta["type"],
                    "path": meta["path"],
                    "cmd_line": meta["cmd_line"],
                    "remote_ip": meta["remote_ip"],
                    "remote_port": meta["remote_port"]
                })
            
            new_v_df = pd.DataFrame(new_v_list)
            combined_v = pd.concat([existing_v, new_v_df], ignore_index=True).drop_duplicates(subset=["node_id"])
            combined_v.to_csv(v_file, index=False)

            # Load or append edges
            if os.path.exists(e_file):
                try:
                    existing_e = pd.read_csv(e_file)
                except Exception:
                    existing_e = pd.DataFrame(columns=["edge_id", "src", "dst", "event", "time"])
            else:
                existing_e = pd.DataFrame(columns=["edge_id", "src", "dst", "event", "time"])

            new_e_df = pd.DataFrame(self.edges_buffer)
            combined_e = pd.concat([existing_e, new_e_df], ignore_index=True)
            combined_e.to_csv(e_file, index=False)

            print(f"[*] Flushed {len(self.edges_buffer)} edges and unique vertices to disk.")
            self.edges_buffer.clear()
            self.vertices_dict.clear()

        try:
            while self.running:
                b.perf_buffer_poll(timeout=100)
                
                # Check duration exit
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
            print("[*] Collector stopped cleanly.")

    def train(self, data_dir, model_dir, train_ratio, val_ratio, epochs, lr):
        os.makedirs(model_dir, exist_ok=True)
        print(f"[*] Starting offline VELOX training using data under: {data_dir}")

        v_file = os.path.join(data_dir, "vertex.csv")
        e_file = os.path.join(data_dir, "edge.csv")

        if not os.path.exists(v_file) or not os.path.exists(e_file):
            print(f"[!] Error: Missing vertex.csv or edge.csv under: {data_dir}")
            sys.exit(1)

        print("[*] Loading vertex and edge datasets...")
        vertex = pd.read_csv(v_file)
        edge = pd.read_csv(e_file)
        print(f"[*] Vertices: {len(vertex)} | Edges: {len(edge)}")

        if len(edge) < 10:
            print("[!] Error: Insufficient data to partition. Need at least 10 edges.")
            sys.exit(1)

        # Chronological sorting for time split
        edge = edge.sort_values('time').reset_index(drop=True)
        min_time = edge['time'].min()
        max_time = edge['time'].max()
        duration = max_time - min_time

        train_end = min_time + duration * train_ratio
        val_end = min_time + duration * (train_ratio + val_ratio)

        train_edge_df = edge[edge['time'] <= train_end].copy()
        val_edge_df = edge[(edge['time'] > train_end) & (edge['time'] <= val_end)].copy()
        test_edge_df = edge[edge['time'] > val_end].copy()

        print(f"[*] Split sizes - Train: {len(train_edge_df)} | Val: {len(val_edge_df)} | Test: {len(test_edge_df)}")

        # Collect corpus of words from train nodes for Word2Vec training
        train_nodes = set(train_edge_df['src'].unique()).union(set(train_edge_df['dst'].unique()))
        train_nodes_str = {str(n) for n in train_nodes}

        corpus = []
        node_id_to_idx = {}
        idx = 0

        for _, row in vertex.iterrows():
            node_id = str(row['node_id'])
            if node_id not in node_id_to_idx:
                node_id_to_idx[node_id] = idx
                idx += 1

            if node_id in train_nodes_str:
                node_type = str(row.get('type', 'unknown'))
                terms = [node_type]
                for field in ['path', 'cmd_line', 'remote_ip', 'remote_port']:
                    if field in row and pd.notna(row[field]):
                        val = str(row[field]).strip()
                        if val:
                            tokens = [t for t in val.replace('/', ' ').split() if t]
                            terms.extend(tokens)
                corpus.append(terms)

        if not corpus:
            print("[!] Error: Word2Vec corpus is empty! No process characteristics to learn.")
            sys.exit(1)

        print(f"[*] Training Word2Vec model on {len(corpus)} process/resource definitions...")
        w2v_model = Word2Vec(
            sentences=corpus,
            vector_size=W2V_DIM,
            window=5,
            min_count=1,
            epochs=100,
            workers=4
        )

        # Construct lookup feature embeddings for all known vertices
        print("[*] Generating feature embeddings for all nodes...")
        x_nodes = np.zeros((len(node_id_to_idx), W2V_DIM), dtype=np.float32)
        node_embeddings_dict = {}

        for _, row in vertex.iterrows():
            node_id = str(row['node_id'])
            node_idx = node_id_to_idx[node_id]

            node_type = str(row.get('type', 'unknown'))
            terms = [node_type]
            for field in ['path', 'cmd_line', 'remote_ip', 'remote_port']:
                if field in row and pd.notna(row[field]):
                    val = str(row[field]).strip()
                    if val:
                        tokens = [t for t in val.replace('/', ' ').split() if t]
                        terms.extend(tokens)

            embeddings = []
            for t in terms:
                if t in w2v_model.wv:
                    embeddings.append(w2v_model.wv[t])
            
            if embeddings:
                vec = np.mean(embeddings, axis=0)
                x_nodes[node_idx] = vec
                node_embeddings_dict[node_id] = vec.tolist()
            else:
                node_embeddings_dict[node_id] = np.zeros(W2V_DIM).tolist()

        # Map edge classes
        event_types = sorted(edge['event'].unique())
        global event_to_idx
        # If event classes differ, align with current mapping
        event_to_idx = {ev: i for i, ev in enumerate(event_types)}
        num_edge_types = len(event_to_idx)

        def create_graph_data(edge_df):
            edge_indices = []
            edge_types = []
            for _, row in edge_df.iterrows():
                src = str(row['src'])
                dst = str(row['dst'])
                if src in node_id_to_idx and dst in node_id_to_idx:
                    edge_indices.append([node_id_to_idx[src], node_id_to_idx[dst]])
                    edge_types.append(event_to_idx[row['event']])

            if not edge_indices:
                return Data(
                    x=torch.tensor(x_nodes, dtype=torch.float),
                    edge_index=torch.empty((2, 0), dtype=torch.long),
                    edge_attr=torch.empty((0, num_edge_types), dtype=torch.float)
                )

            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            e_type = torch.tensor(edge_types, dtype=torch.long)
            edge_attr = F.one_hot(e_type, num_classes=num_edge_types).float()
            x = torch.tensor(x_nodes, dtype=torch.float)
            return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

        print("[*] Generating validation subgraphs...")
        train_data = create_graph_data(train_edge_df).to(DEVICE)
        val_data = create_graph_data(val_edge_df).to(DEVICE)

        # Initialize network model
        print(f"[*] Initializing VeloxModel CustomEdgeMLP on target device: {DEVICE}...")
        model = VeloxModel(W2V_DIM, NODE_OUT_DIM, num_edge_types).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        print(f"[*] Commencing training loop ({epochs} epochs)...")
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            losses = model(train_data)
            if losses.numel() == 0:
                print("[!] Error: No training edges could be compiled into subgraph!")
                sys.exit(1)
            loss = losses.mean()
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"    Epoch {epoch+1:3d}/{epochs} | Training CrossEntropy Loss: {loss.item():.6f}")

        # Compute threshold boundaries (max loss over validation dataset)
        model.eval()
        with torch.no_grad():
            val_losses_t = model(val_data)
            if val_losses_t.numel() > 0:
                val_losses = val_losses_t.cpu().numpy()
                threshold = float(np.max(val_losses))
            else:
                threshold = 10.0  # Safe fallback if val edges are empty
                print("[!] Warning: Empty validation subgraph. Defaulting threshold to 10.0.")

        print(f"[*] Optimized Validation Maximum Loss Threshold: {threshold:.6f}")

        # Serialization save weights and meta configurations
        model_path = os.path.join(model_dir, "velox_model.pth")
        torch.save(model.state_dict(), model_path)

        meta_path = os.path.join(model_dir, "velox_meta.pkl")
        meta_state = {
            "w2v_model": w2v_model,
            "embeddings": node_embeddings_dict,
            "event_to_idx": event_to_idx,
            "threshold": threshold,
            "num_edge_types": num_edge_types
        }
        with open(meta_path, "wb") as f:
            pickle.dump(meta_state, f)

        print(f"[*] VELOX baseline checkpoint files successfully created in: {model_dir}")

    def monitor(self, model_dir, check_interval):
        # Dynamic import for BCC wrapper compatibility
        from bcc import BPF

        signal.signal(signal.SIGINT, self.handle_signal)
        signal.signal(signal.SIGTERM, self.handle_signal)

        model_path = os.path.join(model_dir, "velox_model.pth")
        meta_path = os.path.join(model_dir, "velox_meta.pkl")

        if not os.path.exists(model_path) or not os.path.exists(meta_path):
            print(f"[!] Error: VELOX checkpoint model files missing in: {model_dir}")
            print("[*] Please run the offline 'train' command before running 'monitor'.")
            sys.exit(1)

        print(f"[*] Loading meta configuration structures: {meta_path}")
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)

        w2v_model = meta["w2v_model"]
        trained_embeddings = meta["embeddings"]
        event_to_idx_dict = meta["event_to_idx"]
        threshold = meta["threshold"]
        num_edge_types = meta["num_edge_types"]

        print(f"[*] Loading PyTorch weights tensor: {model_path}")
        model = VeloxModel(W2V_DIM, NODE_OUT_DIM, num_edge_types).to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()

        w2v_cache = {}

        def get_w2v_vector(node_id, node_type, path="", cmd_line="", remote_ip="", remote_port=""):
            if node_id in w2v_cache:
                return w2v_cache[node_id]

            if node_id in trained_embeddings:
                vec = torch.tensor(trained_embeddings[node_id], dtype=torch.float).to(DEVICE)
                w2v_cache[node_id] = vec
                return vec

            # Tokenize new dynamic node terms
            terms = [node_type]
            if path:
                terms.extend([t for t in path.replace('/', ' ').split() if t])
            if cmd_line:
                terms.extend([t for t in cmd_line.replace('/', ' ').split() if t])
            if remote_ip:
                terms.append(remote_ip)
            if remote_port:
                terms.append(str(remote_port))

            embeddings = []
            for t in terms:
                if t in w2v_model.wv:
                    embeddings.append(w2v_model.wv[t])

            if embeddings:
                vec = torch.tensor(np.mean(embeddings, axis=0), dtype=torch.float).to(DEVICE)
            else:
                vec = torch.zeros(W2V_DIM, dtype=torch.float).to(DEVICE)

            w2v_cache[node_id] = vec
            return vec

        # Resolve exclusions
        self_pid = os.getpid()
        bash_ancestor = find_nearest_bash_ancestor(self_pid)
        exclude_set = get_all_descendants(bash_ancestor) if bash_ancestor else {self_pid}

        print(f"[*] Excluding process tree rooted at bash({bash_ancestor}) with {len(exclude_set)} PIDs.")

        print("[*] Starting in-kernel BCC tracepoint compiler...")
        b = BPF(text=BPF_TEXT)

        # Set up active FD maps
        syscall_fd_map = b.get_table("syscall_fd_map")
        for sc_id, f_idx in fd_syscall_map_dict.items():
            syscall_fd_map[ct.c_uint32(sc_id)] = ct.c_int32(f_idx)

        # Set exclusions in kernel map
        ex_map = b.get_table("excluded_pids")
        for p in exclude_set:
            ex_map[ct.c_uint32(p)] = ct.c_uint8(1)

        def handle_event(cpu, data, size):
            event = b["events"].event(data)

            comm_str = event.comm.decode('utf-8', 'replace').strip()
            s_raw = f"{comm_str}:{event.pid}"
            s_uuid_str = str(uuid.uuid5(NS_SUBJECT, s_raw))

            v_name = id_to_event.get(event.syscall_id, "EVENT_UNKNOWN")
            v_idx = event_to_idx_dict.get(v_name, -1)
            if v_idx == -1:
                return # Skip untracked edge events

            # Build subject embedding
            s_vec = get_w2v_vector(
                node_id=s_uuid_str,
                node_type="process",
                cmd_line=comm_str
            )

            # Build object embedding
            if event.is_socket and event.family == socket.AF_INET:
                ip_str = socket.inet_ntoa(struct.pack('<I', event.daddr))
                port_val = socket.ntohs(event.dport)
                o_raw = f"{ip_str}:{port_val}"
                o_uuid_str = str(uuid.uuid5(NS_OBJECT, o_raw))
                o_vec = get_w2v_vector(
                    node_id=o_uuid_str,
                    node_type="socket",
                    remote_ip=ip_str,
                    remote_port=str(port_val)
                )
            else:
                fn_str = event.filename.decode('utf-8', 'replace').strip() or "unknown"
                o_uuid_str = str(uuid.uuid5(NS_OBJECT, fn_str))
                o_vec = get_w2v_vector(
                    node_id=o_uuid_str,
                    node_type="file",
                    path=fn_str
                )

            with torch.no_grad():
                # Consolidate dynamic edge graph indices
                x = torch.stack([s_vec, o_vec])
                edge_index = torch.tensor([[0], [1]], dtype=torch.long).to(DEVICE)
                edge_attr_idx = torch.tensor([v_idx], dtype=torch.long).to(DEVICE)

                loss_tensor = model(x=x, edge_index=edge_index, edge_attr_idx=edge_attr_idx)
                loss_val = float(loss_tensor.cpu().numpy()[0])

            ts_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            if loss_val > threshold:
                print(f"[{ts_str}] [🚨 ALARM] Anomaly! Action: {comm_str}({event.pid}) -> {v_name} -> Object. Loss: {loss_val:.6f} > Threshold: {threshold:.6f}")
            else:
                print(f"[{ts_str}] [INFO] Event benign. Loss: {loss_val:.6f}")

        def handle_lost(lost):
            print(f"\n[!] WARNING: Lost {lost} eBPF events! Buffer capacity overflow.", file=sys.stderr)

        b["events"].open_perf_buffer(handle_event, page_cnt=16384, lost_cb=handle_lost)
        print(f"[*] VELOX monitor engine active on device: {DEVICE}. Listening...")

        try:
            while self.running:
                b.perf_buffer_poll(timeout=int(check_interval * 1000))
        except KeyboardInterrupt:
            pass
        print("[*] VELOX monitor stopped cleanly.")

def main():
    parser = argparse.ArgumentParser(
        description="VELOX Baseline Intrusion Detection Model (eBPF + Word2Vec + EdgeMLP GNN)",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", required=True, help="VELOX baseline subcommands")

    # Collect subcommand
    collect_parser = subparsers.add_parser("collect", help="Hook kernel operations and save CSV logs")
    collect_parser.add_argument("--duration", type=int, default=0, help="Graceful exit timer boundary in seconds (0 = infinite)")
    collect_parser.add_argument("--output-dir", type=str, default="checkpoint/data_lake/velox", help="Target output folder")
    collect_parser.add_argument("--flush-interval", type=float, default=60.0, help="Write log append cycles (seconds)")

    # Train subcommand
    train_parser = subparsers.add_parser("train", help="Optimise network MLP parameters using offline graph logs")
    train_parser.add_argument("--data-dir", type=str, default="checkpoint/data_lake/velox", help="Source dataset folder")
    train_parser.add_argument("--model-dir", type=str, default="checkpoint/velox", help="Target serialization folder")
    train_parser.add_argument("--train-ratio", type=float, default=0.8, help="Ratio of dataset allocated to model optimization")
    train_parser.add_argument("--val-ratio", type=float, default=0.1, help="Ratio of dataset allocated to threshold boundaries check")
    train_parser.add_argument("--epochs", type=int, default=100, help="Total epoch optimization iterations")
    train_parser.add_argument("--lr", type=float, default=0.001, help="Adam optimization step learning rates")

    # Monitor subcommand
    monitor_parser = subparsers.add_parser("monitor", help="Attach live BCC probe and run Real-Time scoring anomalies checks")
    monitor_parser.add_argument("--model-dir", type=str, default="checkpoint/velox", help="Source serialization weights folder")
    monitor_parser.add_argument("--interval", type=float, default=1.0, help="Performance polling checks intervals (seconds)")

    args = parser.parse_args()
    pipeline = VeloxPipeline()

    if args.command == "collect":
        pipeline.collect(duration=args.duration, output_dir=args.output_dir, flush_interval=args.flush_interval)
    elif args.command == "train":
        pipeline.train(
            data_dir=args.data_dir,
            model_dir=args.model_dir,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            epochs=args.epochs,
            lr=args.lr
        )
    elif args.command == "monitor":
        pipeline.monitor(model_dir=args.model_dir, check_interval=args.interval)

if __name__ == "__main__":
    main()
