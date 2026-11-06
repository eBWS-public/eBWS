# -*- coding: utf-8 -*-
import os
import sys
import ast
import json
import ctypes as ct

for _sys_pkg_path in ("/usr/lib/python3/dist-packages", "/usr/local/lib/python3/dist-packages"):
    if os.path.isdir(_sys_pkg_path) and _sys_pkg_path not in sys.path:
        sys.path.append(_sys_pkg_path)

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

# Get clock ticks per second (usually 100)
try:
    USER_HZ = os.sysconf(os.sysconf_names["SC_CLK_TCK"])
except (AttributeError, ValueError, KeyError):
    USER_HZ = 100


def _parse_simple_yaml_value(val_str):
    """Parses scalar or inline list values for the fallback YAML loader."""
    val_str = val_str.strip()
    if not val_str or val_str.lower() in ("null", "none", "~"):
        return None
    if val_str.lower() == "true":
        return True
    if val_str.lower() == "false":
        return False
    try:
        return ast.literal_eval(val_str)
    except Exception:
        return val_str.strip("\"'")


def _fallback_yaml_load(text):
    """Lightweight YAML parser for flat key-value and list configurations when PyYAML is absent."""
    result = {}
    current_list_key = None
    for raw_line in text.splitlines():
        # Strip comments outside quotes
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        stripped = line.lstrip()
        if stripped.startswith("- ") and current_list_key is not None:
            item_val = _parse_simple_yaml_value(stripped[2:])
            result[current_list_key].append(item_val)
            continue
        if ":" in stripped:
            key, val_part = stripped.split(":", 1)
            key = key.strip()
            val_part = val_part.strip()
            if val_part == "":
                result[key] = []
                current_list_key = key
            else:
                result[key] = _parse_simple_yaml_value(val_part)
                current_list_key = None
    return result


def load_config(config_path="config.yaml"):
    """Loads configuration from a YAML file (with a built-in fallback if PyYAML is unavailable)."""
    if not config_path or not os.path.exists(config_path):
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        content = f.read()
    if HAS_YAML:
        cfg = yaml.safe_load(content)
        return cfg if isinstance(cfg, dict) else {}
    return _fallback_yaml_load(content)


def save_config(config, config_path="config.yaml"):
    """Saves configuration dictionary to a YAML file."""
    with open(config_path, "w", encoding="utf-8") as f:
        if HAS_YAML:
            yaml.safe_dump(config, f, sort_keys=False)
        else:
            for k, v in config.items():
                if isinstance(v, list) and all(isinstance(x, str) for x in v):
                    f.write(f"{k}:\n")
                    for item in v:
                        f.write(f'  - "{item}"\n')
                else:
                    f.write(f"{k}: {json.dumps(v)}\n")


def setup_bcc_paths(config=None):
    """Appends configured system package paths so system-installed BCC can be imported inside a venv."""
    default_paths = [
        "/usr/lib/python3/dist-packages",
        "/usr/local/lib/python3/dist-packages",
    ]
    paths = (config or {}).get("system_package_paths", default_paths)
    for p in list(paths) + default_paths:
        if p and p not in sys.path:
            sys.path.append(p)


def fix_sudo_ownership(path):
    """Restores ownership of files/directories created under sudo to the invoking user."""
    try:
        if os.geteuid() == 0 and "SUDO_UID" in os.environ and "SUDO_GID" in os.environ:
            uid = int(os.environ["SUDO_UID"])
            gid = int(os.environ["SUDO_GID"])
            if os.path.exists(path):
                os.chown(path, uid, gid)
    except Exception:
        pass


def read_proc_stat(pid):
    """Reads PPID and start time from /proc/{pid}/stat."""
    path = f"/proc/{pid}/stat"
    with open(path, "r") as f:
        s = f.read().strip()
    rparen = s.rfind(")")
    if rparen == -1:
        raise ValueError(f"Malformed stat for pid={pid}")
    after = s[rparen + 2:].split()
    ppid = int(after[1])
    # Field 22 (starttime) is index 19 in fields after the command name parenthesis
    starttime_ticks = int(after[19])
    starttime_ns = int((starttime_ticks / USER_HZ) * 1e9)
    return ppid, starttime_ns


def read_proc_comm(pid):
    """Reads comm (process name) from /proc/{pid}/comm."""
    path = f"/proc/{pid}/comm"
    with open(path, "r") as f:
        return f.read().strip()


def get_cmdline(pid):
    """Safely retrieves the full command line or process name for a PID."""
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            content = f.read().replace(b"\x00", b" ").strip().decode("utf-8", errors="replace")
        if not content:
            content = read_proc_comm(pid)
        return content if content else "UNKNOWN"
    except Exception:
        return "UNKNOWN"


def get_ancestor_chain(pid):
    """Reconstructs the ancestor chain from proc stat up to init."""
    chain = []
    visited = set()
    cur = pid
    while True:
        if cur in visited:
            break
        visited.add(cur)
        try:
            comm = read_proc_comm(cur)
            chain.append((cur, comm))
        except Exception:
            chain.append((cur, "?"))
            break

        try:
            ppid, _ = read_proc_stat(cur)
        except Exception:
            break

        if ppid == 0:
            break
        cur = ppid
    return chain


def find_nearest_bash_ancestor(pid):
    """Finds the nearest bash ancestor PID."""
    chain = get_ancestor_chain(pid)
    for p, comm in chain:
        if comm == "bash":
            return p
    return None


def get_all_processes():
    """Gets list of all running processes with PPID and comm."""
    proc_info = {}
    if not os.path.exists("/proc"):
        return proc_info
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        pid = int(name)
        try:
            ppid, _ = read_proc_stat(pid)
            comm = read_proc_comm(pid)
            proc_info[pid] = {"ppid": ppid, "comm": comm}
        except Exception:
            continue
    return proc_info


def build_children_map(proc_info):
    """Builds an adjacency list of PPID -> children PIDs."""
    children = {}
    for pid, info in proc_info.items():
        ppid = info["ppid"]
        children.setdefault(ppid, []).append(pid)
    return children


def get_descendants(root_pid, children_map):
    """Performs DFS/BFS to retrieve all descendant PIDs from a root PID."""
    result = set()
    stack = [root_pid]
    while stack:
        cur = stack.pop()
        if cur in result:
            continue
        result.add(cur)
        for ch in children_map.get(cur, []):
            if ch not in result:
                stack.append(ch)
    return result


def get_excluded_pids():
    """Gets the set of PIDs that should be excluded from monitoring."""
    self_pid = os.getpid()
    exclude_root_pid = find_nearest_bash_ancestor(self_pid)

    proc_info = get_all_processes()
    children_map = build_children_map(proc_info)

    if exclude_root_pid:
        excluded = get_descendants(exclude_root_pid, children_map)
    else:
        # Fallback to self and descendants if no bash ancestor is found
        exclude_root_pid = self_pid
        excluded = get_descendants(exclude_root_pid, children_map)
    return excluded


def load_excluded_pids_to_bpf(bpf_obj, pid_set):
    """Loads excluded PID set into eBPF excluded_pids map."""
    excluded = bpf_obj["excluded_pids"]
    one = ct.c_uint8(1)
    for pid in sorted(pid_set):
        excluded[ct.c_uint32(pid)] = one
