# -*- coding: utf-8 -*-
from collections import OrderedDict


class PLNode:
    """Represents a node in the Process Lineage Tree."""
    def __init__(self, pid, start_t, ppid, pstart_t, comm, time_val):
        self.pid = pid
        self.start_t = start_t
        self.ppid = ppid
        self.pstart_t = pstart_t
        self.comm = comm
        self.time = time_val  # last system call timestamp, or -1 if only traversed

    def __repr__(self):
        return f"PLNode(pid={self.pid}, comm={self.comm}, time={self.time})"


class PLTree:
    """Manages parent-child process lineage with strict LRU cache eviction."""
    def __init__(self, max_size=50000):
        self.max_size = int(max_size) if (max_size is not None and int(max_size) > 0) else None
        self.nodes = OrderedDict()  # (pid, start_t) -> PLNode
        self.children = {}          # (ppid, pstart_t) -> set of (pid, start_t)
        self.eviction_count = 0
        self.peak_size = 0

    def __len__(self):
        return len(self.nodes)

    def record(self, pid, start_t, ppid, pstart_t, comm, time_val):
        """Records or updates a node, and maintains parent-child mappings."""
        key = (pid, start_t)
        parent_key = (ppid, pstart_t) if ppid > 0 else None

        if key in self.nodes:
            node = self.nodes[key]
            # Update timestamp if the new timestamp is a real system call time
            if time_val != -1 or node.time == -1:
                node.time = time_val

            # If parent information is provided and differs, update parent-child maps
            if ppid > 0:
                old_parent_key = (node.ppid, node.pstart_t) if node.ppid > 0 else None
                if old_parent_key and old_parent_key != parent_key:
                    if old_parent_key in self.children:
                        self.children[old_parent_key].discard(key)

                node.ppid = ppid
                node.pstart_t = pstart_t
                if parent_key:
                    self.children.setdefault(parent_key, set()).add(key)

            self.nodes.move_to_end(key)
        else:
            node = PLNode(pid, start_t, ppid, pstart_t, comm, time_val)
            self.nodes[key] = node
            if parent_key:
                self.children.setdefault(parent_key, set()).add(key)

        self.evict_old()
        if len(self.nodes) > self.peak_size:
            self.peak_size = len(self.nodes)
        return node

    def get_node(self, pid, start_t):
        """Gets a node if it exists and moves it to the end of the LRU cache."""
        key = (pid, start_t)
        if key in self.nodes:
            self.nodes.move_to_end(key)
            return self.nodes[key]
        return None

    def reconstruct_chain(self, pid, start_t):
        """Reconstructs the lineage tree chain back to init or root ancestor."""
        chain = []
        visited = set()
        curr_key = (pid, start_t)

        while curr_key and curr_key not in visited:
            visited.add(curr_key)
            node = self.nodes.get(curr_key)
            if not node:
                # Fallback to search by PID alone if start_t is not exact or unknown
                found = False
                for k, n in self.nodes.items():
                    if k[0] == curr_key[0]:
                        node = n
                        curr_key = k
                        found = True
                        break
                if not found:
                    break

            chain.append(node)
            if node.ppid <= 0 or node.ppid == 1:
                break
            curr_key = (node.ppid, node.pstart_t)

        return chain

    def evict_old(self):
        """Evicts old nodes while keeping total size <= max_size.
        Strictly preserves nodes added through upstream traversal (time = -1)
        if they still have active child nodes in the tree.
        """
        if self.max_size is None:
            return

        excess = len(self.nodes) - self.max_size
        if excess <= 0:
            return

        keys_to_evict = []
        current_keys = list(self.nodes.keys())
        idx = 0

        while excess > 0 and idx < len(current_keys):
            key = current_keys[idx]
            node = self.nodes[key]

            # Check if this node has active children in the tree
            has_active_children = False
            if key in self.children and self.children[key]:
                # Clean up any child keys that have been evicted previously
                active_children = {ch for ch in self.children[key] if ch in self.nodes}
                self.children[key] = active_children
                if active_children:
                    has_active_children = True

            # If node was created via traversal (time = -1) and still has children, skip it
            if node.time == -1 and has_active_children:
                idx += 1
                continue

            # Eligible for eviction
            keys_to_evict.append(key)
            excess -= 1
            idx += 1

        for key in keys_to_evict:
            node = self.nodes.pop(key)
            self.eviction_count += 1
            # Remove from parent's child tracking
            parent_key = (node.ppid, node.pstart_t) if node.ppid > 0 else None
            if parent_key and parent_key in self.children:
                self.children[parent_key].discard(key)
            # Clean up child tracking for this key
            if key in self.children:
                del self.children[key]

    def get_stats(self):
        """Returns diagnostic statistics of the Process Lineage Tree."""
        locked_ancestors = 0
        for key, node in self.nodes.items():
            if node.time == -1 and key in self.children:
                if any(ch in self.nodes for ch in self.children[key]):
                    locked_ancestors += 1
        return {
            "current_size": len(self.nodes),
            "peak_size": self.peak_size,
            "max_size": self.max_size,
            "eviction_count": self.eviction_count,
            "locked_ancestors": locked_ancestors,
        }
