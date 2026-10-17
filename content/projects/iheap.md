---
title: "iheap"
description: "An indexed priority queue for Python with a Rust core. A drop-in-style replacement for pqdict that runs 3–20× faster, with O(log n) priority updates and deletes for any item."
summary: "Fast indexed priority queue for Python with a Rust core"
tags: ["rust", "python", "data-structures"]
externalUrl: "https://github.com/nvbkdw/iheap"
weight: 5
hidemeta: true
---

An indexed priority queue for Python with a Rust core. Unlike the standard `heapq`, it can change the priority of any item, or remove any item, in O(log n). That makes it useful for Dijkstra/A* search, task schedulers, LRU/LFU-style caches, event simulation, and top-k tracking. It is a drop-in-style replacement for [`pqdict`](https://github.com/nvictus/pqdict) and runs 3–20× faster.

```python
from iheap import PQDict

pq = PQDict({"a": 3, "b": 1, "c": 2})  # key -> priority, min on top
pq["d"] = 0                            # insert
pq["a"] = -1                           # update priority
del pq["c"]                            # delete
assert pq.top() == "a"
```

## Key Features

- **Full dict semantics** — `PQDict` supports `get`, `setdefault`, `|`/`|=`, views, insertion-order iteration, and pickling, on top of the pqdict priority-queue API
- **Payloads stay in Python** — only float priorities and integer handles cross into Rust; keys and values are never copied or converted
- **Fast core** — a cache-friendly 4-ary heap with O(n) bulk construction and bulk updates in a single native call; core operations take roughly 5–70 ns
- **Two API levels** — high-level `PQDict`, or the handle-based `IndexedHeap` when you manage your own storage
- **Easy install** — prebuilt `abi3` wheels for CPython 3.10+ on Linux, macOS, and Windows, with type information (`py.typed`)

**Tech stack**: Rust, Python (native extension, abi3 wheels)

[View documentation](https://nvbkdw.github.io/iheap/) · [View on GitHub](https://github.com/nvbkdw/iheap)
