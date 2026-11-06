#include <uapi/linux/ptrace.h>
#include <linux/sched.h>

struct key_t {
    u32 pid;
    u64 start_t;
    u32 ppid;
    u64 pstart_t;
    u32 syscall_id;
};

// Set of PIDs to exclude from monitoring (e.g. self and parent bash subtree)
BPF_HASH(excluded_pids, u32, u8, 131072);

// Double buffering count maps
BPF_PERCPU_HASH(counts_0, struct key_t, u64, 65536);
BPF_PERCPU_HASH(counts_1, struct key_t, u64, 65536);

// Control map: holding 0 or 1 to select the active count map
BPF_ARRAY(control_map, u32, 1);

TRACEPOINT_PROBE(raw_syscalls, sys_enter) {
    u64 id = bpf_get_current_pid_tgid();
    u32 tgid = id >> 32;

    // 1. Check if process is excluded
    u8 *excluded = excluded_pids.lookup(&tgid);
    if (excluded) {
        return 0;
    }

    // 2. Fetch the active map index (0 or 1)
    u32 index = 0;
    u32 map_idx_key = 0;
    u32 *idx_ptr = control_map.lookup(&map_idx_key);
    if (idx_ptr) {
        index = *idx_ptr;
    }

    // 3. Populate composite process event key
    struct key_t key = {};
    struct task_struct *task = (struct task_struct *)bpf_get_current_task();
    
    key.pid = tgid;
    key.start_t = task->start_time;

    struct task_struct *parent = task->real_parent;
    if (parent) {
        key.ppid = parent->tgid;
        key.pstart_t = parent->start_time;
    } else {
        key.ppid = 0;
        key.pstart_t = 0;
    }

    key.syscall_id = args->id;

    u64 zero = 0;
    u64 *val;

    if (index == 0) {
        val = counts_0.lookup_or_try_init(&key, &zero);
    } else {
        val = counts_1.lookup_or_try_init(&key, &zero);
    }

    if (val) {
        (*val)++;
    }

    return 0;
}
