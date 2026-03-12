# SGLang HiCache: External KV Cache Storage Integration with the Scheduler

## 1. The Big Picture: Three-Tier KV Cache Hierarchy

SGLang's HiCache implements a **three-tier memory hierarchy** for KV cache:

```
┌─────────────────────┐
│   L1: GPU (Device)  │  ← Hot. Used directly by attention kernels.
│   TokenToKVPool     │    Fastest, but smallest.
├─────────────────────┤
│   L2: CPU (Host)    │  ← Warm. Pinned host memory.
│   HostKVCache       │    ~2× GPU capacity. Async GPU↔CPU DMA.
├─────────────────────┤
│   L3: Storage       │  ← Cold. Disk, RDMA, distributed stores.
│   HiCacheStorage    │    Largest. Async prefetch/backup threads.
└─────────────────────┘
```

The key insight: **evicted GPU nodes don't disappear**. They survive in the radix tree with `value=None` but `host_value` intact, enabling transparent load-back when the same prefix is requested again.

---

## 2. Key Classes and Their Roles

| Class | File | Role |
|---|---|---|
| `Scheduler` | `managers/scheduler.py` | Orchestrator. Triggers prefetch, scheduling, eviction. |
| `HiRadixCache` | `mem_cache/hiradix_cache.py` | Extends `RadixCache` with L2/L3 awareness. The "brain" of the hierarchy. |
| `HiCacheController` | `managers/cache_controller.py` | Async transfer engine. Manages CUDA streams, background threads. |
| `HiCacheStorage` | `mem_cache/hicache_storage.py` | Abstract interface for L3 backends (file, Mooncake, NIXL, etc.) |
| `TreeNode` | `mem_cache/radix_cache.py` | Radix tree node. Carries `value` (GPU), `host_value` (CPU), `hash_value` (storage keys). |
| `SchedulePolicy` / `PrefillAdder` | `managers/schedule_policy.py` | Cache-aware scheduling. Triggers load-back during batch assembly. |
| `LayerDoneCounter` | `managers/cache_controller.py` | Per-layer sync ring buffer enabling overlap of loading and forward. |
| `HostKVCache` | `mem_cache/memory_pool_host.py` | CPU pinned memory pool with layout variants (layer_first, page_first, etc.) |
| `StorageBackendFactory` | `mem_cache/storage/backend_factory.py` | Lazy-loading factory for L3 backends with dynamic plugin support. |

---

## 3. The Radix Tree: Extended for Three Tiers

Each `TreeNode` in the base `RadixCache` was originally:

```python
node.key    = RadixKey(token_ids)  # edge label
node.value  = torch.Tensor         # GPU KV cache indices
```

`HiRadixCache` extends this to:

```python
node.value            = torch.Tensor | None   # GPU indices (None if evicted from GPU)
node.host_value       = torch.Tensor | None   # CPU indices (None if not backed up)
node.hash_value       = List[str]             # SHA256 per page (storage addressing key)
node.host_ref_counter = int                   # protection from host eviction during async I/O
node.lock_ref         = int                   # protection from GPU eviction
node.hit_count        = int                   # access frequency, triggers write-through
```

### Node Lifecycle

```
Created → value=GPU_indices, host_value=None
  ↓ (write-through triggered by hit_count >= threshold)
Backed up → value=GPU_indices, host_value=CPU_indices
  ↓ (GPU eviction under memory pressure)
GPU-evicted → value=None, host_value=CPU_indices  [NODE STAYS IN TREE]
  ↓ (host eviction under memory pressure, but data persisted to storage)
Fully evicted → node deleted from tree  [DATA LIVES ONLY IN STORAGE]
```

Key properties on `TreeNode`:
- `node.evicted` → `self.value is None` (GPU data gone)
- `node.backuped` → `self.host_value is not None` (CPU data present)

---

## 4. Initialization: How the Scheduler Creates HiCache

### Configuration Flags (server_args.py)

```
--enable-hierarchical-cache          # Master switch
--hicache-ratio 2.0                  # Host pool = 2× GPU pool
--hicache-size 0                     # Host pool in GB (overrides ratio if >0)
--hicache-write-policy write_through # write_through | write_through_selective | write_back
--hicache-io-backend kernel          # direct | kernel | kernel_ascend
--hicache-mem-layout layer_first     # layer_first | page_first | page_first_direct | page_head
--hicache-storage-backend file       # file | mooncake | hf3fs | nixl | aibrix | eic | dynamic
--hicache-storage-prefetch-policy best_effort  # best_effort | wait_complete | timeout
--hicache-storage-backend-extra-config '{}'    # JSON or @filepath
```

### Scheduler Initialization (scheduler.py:636)

```python
def init_cache_with_memory_pool(self):
    # Two boolean flags control the feature:
    self.enable_hierarchical_cache = server_args.enable_hierarchical_cache
    self.enable_hicache_storage = server_args.hicache_storage_backend is not None

    # ... memory pool setup ...

    if self.enable_hierarchical_cache:
        from sglang.srt.mem_cache.hiradix_cache import HiRadixCache
        self.tree_cache = HiRadixCache(params=params, server_args=server_args)

        # Connect per-layer sync counter to the model worker
        self.tp_worker.register_hicache_layer_transfer_counter(
            self.tree_cache.cache_controller.layer_done_counter
        )
```

### HiRadixCache.__init__ (hiradix_cache.py:53)

1. Creates the host KV pool (`MHATokenToKVPoolHost` / `MLATokenToKVPoolHost` / `NSATokenToKVPoolHost`)
2. Creates the `HiCacheController` with device allocator, host pool, write/load CUDA streams
3. Optionally attaches a storage backend via `StorageBackendFactory`
4. Initializes tracking dictionaries:
   - `ongoing_write_through: Dict[int, TreeNode]`
   - `ongoing_load_back: Dict[int, TreeNode]`
   - `ongoing_prefetch: Dict[str, tuple]`
   - `ongoing_backup: Dict[int, TreeNode]`

---

## 5. Integration Points: The Complete Data Flow

### Phase 1: Request Arrival → Storage Prefetch

```
handle_generate_request(recv_req)           # scheduler.py:1481
  └→ _add_request_to_queue(req)             # scheduler.py:1682
      └→ _prefetch_kvcache(req)             # scheduler.py:1659
```

```python
def _prefetch_kvcache(self, req: Req):
    if self.enable_hicache_storage:
        # 1. Run prefix match against the radix tree
        req.init_next_round_input(self.tree_cache)

        # 2. If the deepest match has host backup, there might be MORE in storage
        if req.last_node.backuped:
            last_hash = req.last_host_node.get_last_hash_value()
            matched_len = len(req.prefix_indices) + req.host_hit_length
            new_input_tokens = req.fill_ids[matched_len:]

            # 3. Kick off async storage → host prefetch
            self.tree_cache.prefetch_from_storage(
                req.rid, req.last_host_node, new_input_tokens, last_hash, ...
            )
```

#### Step-by-Step Walkthrough

**Step 1: `req.init_next_round_input(self.tree_cache)`** (schedule_batch.py:888)

Builds `fill_ids = origin_input_ids + output_ids` (the complete token sequence), then calls `tree_cache.match_prefix()` to walk the radix tree. The match populates five fields on the request:

| Field | Type | Meaning |
|---|---|---|
| `req.prefix_indices` | `torch.Tensor` | GPU KV pool indices for the matched prefix |
| `req.last_node` | `TreeNode` | Deepest node with GPU data (`value != None`) |
| `req.last_host_node` | `TreeNode` | Deepest node with CPU data (`host_value != None`) — may extend deeper than `last_node` |
| `req.host_hit_length` | `int` | Extra tokens found in CPU beyond the GPU match |
| `req.fill_ids` | `List[int]` | Complete token sequence |

```
fill_ids:  [tok0 ... tok63 | tok64 ... tok127 | tok128 ... tok191 | tok192 ... tok255]
            └── GPU hit ────┘  └── CPU hit ─────┘  └───── uncached ─────────────────────┘
            prefix_indices      host_hit_length      new_input_tokens → goes to storage
```

**Step 2: `if req.last_node.backuped`** (radix_cache.py:129)

```python
@property
def backuped(self):
    return self.host_value is not None
```

If the last GPU node has **no** host backup, the GPU data is the **only** copy — we can't safely assume storage has anything beyond this point, and the node's GPU memory must stay locked. If it **does** have a host backup, storage might have **even more** data beyond what CPU has cached.

**Step 3: Calculate what's uncached**

- **`last_hash`** — SHA256 hash of the last page in the deepest host-backed node. This is the **continuation point** for storage: "start fetching pages that chain after this hash."
- **`matched_len`** — Total tokens already available across GPU + CPU. Everything before this is already cached.
- **`new_input_tokens`** — Remaining tokens that exist **nowhere** in the cache hierarchy. These are what storage is asked to look up.

**Step 4: Prefix keys (optional)**

```python
prefix_keys = (
    req.last_node.get_prefix_hash_values(req.last_node.parent)
    if self.tree_cache.hicache_storage_pass_prefix_keys
    else None
)
```

`get_prefix_hash_values` (radix_cache.py:150) recursively collects all page hashes from root to the node. Some storage backends (e.g., Mooncake, NIXL) need the full hash chain to reconstruct the sequence context.

**Step 5: `prefetch_from_storage`** (hiradix_cache.py:1163)

Kicks off async storage → host I/O. Detailed below.

---

**Why prefetch at enqueue time?** Storage I/O is slow. By starting prefetch when the request *enters the queue* (not when it's scheduled), you maximize lead time. By the time `_get_new_batch_prefill_raw` picks the request up, the storage data has (hopefully) already arrived in host memory.

#### Inside `prefetch_from_storage` (hiradix_cache.py:1163)

1. Aligns tokens to page boundaries (truncates remainder)
2. Checks thresholds and rate limits (skip if storage disabled, too few tokens, or rate-limited)
3. Protects the host node (`host_ref_counter++` to prevent eviction during I/O)
4. Allocates host memory (evicting host if needed)
5. Sends to `HiCacheController.prefetch()` → background thread
6. Tracks the operation in `ongoing_prefetch[req_id]` for later integration

#### Two-Stage Prefetch Pipeline

```
Stage 1 (prefetch_thread):                Stage 2 (prefetch_io_aux_thread):
  Dequeue PrefetchOperation                Dequeue from prefetch_buffer
  → _storage_hit_query():                  → _page_transfer():
     Hash tokens per page (SHA256)            Batch-read from storage backend
     Call batch_exists() on backend           Write directly into host memory
     TP all_reduce MIN (sync hit count)       Atomically track completed_tokens
  → If enough hits: push to Stage 2
  → If not: revoke (free host mem)
```

#### How Storage Backends Use Page Hashes

##### The Hash Chain — Content-Addressed Pages

The foundation is a **chained SHA256 hash** computed per page of tokens (`hicache_storage.py:16`):

```python
def get_hash_str(token_ids, prior_hash=None):
    hasher = hashlib.sha256()
    if prior_hash:
        hasher.update(bytes.fromhex(prior_hash))   # chain from previous page
    for t in token_ids:
        hasher.update(t.to_bytes(4, "little"))
    return hasher.hexdigest()
```

Each page's hash **incorporates the previous page's hash**, so the same tokens at different positions produce different keys:

```
Page 0: hash_0 = SHA256(tokens[0:64])
Page 1: hash_1 = SHA256(hash_0 || tokens[64:128])
Page 2: hash_2 = SHA256(hash_1 || tokens[128:192])
```

This makes each hash a **unique fingerprint** of the entire prefix up to that page — critical because KV cache values are position-dependent.

Hash values are computed when a new node is inserted into the radix tree (`hiradix_cache.py`):

```python
if self.enable_storage:
    new_node.hash_value = compute_node_hash_values(new_node, self.page_size)
```

`compute_node_hash_values` (`radix_cache.py:201`) walks the node's token pages, seeding each hash from the parent's last hash to maintain the chain across tree edges.

##### Stage 1 Detail: `_storage_hit_query` (cache_controller.py:878)

```python
def _storage_hit_query(self, operation):
    last_hash = operation.last_hash          # continuation point from host cache
    for each batch of pages:
        for each page of tokens:
            last_hash = get_hash_str(page_tokens, last_hash)   # extend the chain
            batch_hashes.append(last_hash)
        hit_count = storage_backend.batch_exists(batch_hashes)  # how many consecutive pages exist?
        hash_value.extend(batch_hashes[:hit_count])
        if hit_count < len(batch_hashes):
            break                                                # stop at first miss
    return hash_value, total_hit_tokens
```

`batch_exists` returns the number of **consecutive** existing pages. It stops at the first miss because page N+1's KV cache is only meaningful if pages 0..N also exist (the hash chain enforces this sequential dependency).

##### Stage 2 Detail: `_page_transfer` (cache_controller.py:828)

```python
def _page_transfer(self, operation):
    for each batch of hit hashes:
        storage_backend.batch_get_v1(batch_hashes, batch_host_indices)
        # → storage reads KV data directly into pre-allocated host memory
        operation.increment(completed_tokens)
```

The zero-copy path (`batch_get_v1`) passes host memory pointers directly to the backend, avoiding intermediate copies.

##### Backend Key Transformation

Each backend receives the same SHA256 hash keys but transforms them with TP-awareness suffixes:

| Backend | Key Format | Example |
|---|---|---|
| File | `{hash}_{model}_{tp_rank}_{tp_size}.bin` | `a3f8...e7b1_llama_0_4.bin` |
| Mooncake (MHA) | `{hash}_{rank}_{world_size}_k`, `..._v` | `a3f8...e7b1_0_4_k` |
| Mooncake (MLA) | `{hash}_{pp_rank}_k` | `a3f8...e7b1_0_k` |
| NIXL | `{hash}_{model}_{tp_rank}_{tp_size}_k`, `..._v` | `a3f8...e7b1_llama_0_4_k` |
| HF3FS | `{hash}` → metadata service → page index | `a3f8...e7b1` → index 4217 |

MHA models store separate K and V tensors (two keys per page). MLA models share a single fused KV representation (one key per page, and only rank 0 writes to storage since all TP workers have equivalent state).

##### Per-Backend Storage Mapping

**File backend** (`hicache_storage.py:185`) — one `.bin` file per page hash:

```
hash key → /tmp/hicache/{suffixed_key}.bin
exists() → os.path.exists(path)
get()    → open(path, "rb").readinto(host_tensor)
set()    → tensor.numpy().tofile(path)
```

**Mooncake** (`mooncake_store.py`) — RDMA-based distributed KV store:

```
hash key → suffixed store key
exists() → store.batch_exists(key_strs)
get()    → store.batch_get_into(keys, buffer_ptrs, sizes)   ← zero-copy RDMA read
set()    → batch_exists first (deduplicate), then store.batch_put_from(keys, ptrs, sizes)
```

The RDMA zero-copy path means the storage engine reads/writes directly from/to pinned host memory — no intermediate copies.

**NIXL** (`hicache_nixl.py`) — transfer engine abstraction supporting FILE and OBJ memory types:

```
hash key → suffixed key
exists() → agent.query_memory(tuples, backend_name)
get()    → FILE mode: file_manager.get_file_path(key) → NIXL READ transfer
           OBJ mode:  NIXL READ transfer using key as object identifier
set()    → FILE mode: file_manager.create_file(path) → NIXL WRITE transfer
           OBJ mode:  NIXL WRITE transfer using key as object identifier
```

**HF3FS** (`storage_hf3fs.py`) — metadata service maps hashes to page indices in a shared file:

```
hash key → metadata_client.get_page_indices() → page_index
file offset = page_index × bytes_per_page
get() → usrbio kernel-bypass read at offset
set() → metadata_client.reserve_and_allocate_page_indices() → write at offset → confirm_write()
```

The metadata service provides centralized coordination — multiple TP workers share the same HF3FS file without conflicts.

##### Hash Key as Universal Address

```
SHA256 hash chain (position-aware)
        │
        ├── File backend:     hash → filesystem path → read/write .bin file
        ├── Mooncake:         hash → RDMA KV store key → zero-copy DMA
        ├── NIXL:             hash → FILE path or OBJ key → transfer engine
        └── HF3FS:            hash → metadata service → page index → file offset
```

The hash serves as a **content-addressed, position-aware, distributed identifier**. Every backend receives the same hash keys — they only differ in how they map those keys to physical storage. This abstraction lets you swap backends without changing any cache logic.

**How to fetch multiple subsequent pages from storage backend?**
prefetch thread already knows the full token sequence from the request itself (new_input_tokens). It uses those tokens to compute the expected hash keys    
locally, then asks storage: "do these keys exist?"

Here's the flow:

Request arrives with: [tok0, tok1, tok2, ..., tok191]

After prefix matching:
```
  GPU has:  [tok0 ... tok63]     → prefix_indices
  CPU has:  [tok64 ... tok127]   → host_hit_length
  Unknown:  [tok128 ... tok191]  → new_input_tokens  (passed to prefetch)

  last_hash = hash of the last host-cached page (tok64..tok127's hash)
```

The prefetch thread receives new_input_tokens = [tok128 ... tok191] and last_hash. It can compute every subsequent page hash without asking storage anything:

```python
# _storage_hit_query (cache_controller.py:878)
def _storage_hit_query(self, operation):
    last_hash = operation.last_hash          # hash of tok64..127 (from CPU cache)
    tokens = operation.token_ids             # [tok128, tok129, ..., tok191]

    # Compute hashes locally from the request's own tokens
    for each page of tokens:
        last_hash = get_hash_str(page_tokens, last_hash)
        #   page 0: hash_2 = SHA256(last_hash || tok128..tok191)
        batch_hashes.append(last_hash)

    # NOW ask storage: "do you have data for these specific hashes?"
    hit_count = storage_backend.batch_exists(batch_hashes)
    #   → returns number of consecutive pages that exist
    #   → e.g., 1 means storage has page for hash_2 but NOT hash_3
```

### Phase 2: Scheduling → Check Prefetch → Load Back

When the scheduler builds the next prefill batch (`_get_new_batch_prefill_raw`, scheduler.py:1980):

```python
# A. Process async HiCache events (completions)
if self.enable_hierarchical_cache:
    self.tree_cache.check_hicache_events()        # scheduler.py:2012

# B. Calculate scheduling priorities (runs match_prefix per req)
self.policy.calc_priority(self.waiting_queue, ...)

# C. For each candidate request:
for req in self.waiting_queue:
    if self.enable_hicache_storage:
        # C1. Check if prefetch completed
        prefetch_done = self.tree_cache.check_prefetch_progress(req.rid)
        if not prefetch_done:
            continue  # Skip — still fetching from storage
        req.storage_hit_length = self.tree_cache.pop_prefetch_loaded_tokens(req.rid)

    # C2. Re-match prefix (now includes prefetched host data)
    req.init_next_round_input(self.tree_cache)

    # C3. PrefillAdder decides how many tokens to compute
    result = adder.add_one_req(req, ...)
```

**`check_hicache_events()`** (hiradix_cache.py:960) processes three async completion queues:

| Queue | What it processes |
|---|---|
| `writing_check()` | GPU→Host write-through completions → triggers storage backup |
| `loading_check()` | Host→GPU load-back completions → unlocks nodes |
| `drain_storage_control_queues()` | Prefetch revocations, backup acks, host mem releases |

All synchronized across TP workers via `all_reduce(MIN)`.

### Phase 3: PrefillAdder → Load-Back from Host

Inside `PrefillAdder.add_one_req` (schedule_policy.py:719):

```python
# Account for tokens that will be loaded from host (not recomputed)
real_input_tokens = req.extend_input_len - req.host_hit_length

if req.host_hit_length > 0:
    # Trigger async Host → GPU copy
    new_indices, req.last_node = self.tree_cache.init_load_back(
        req.last_host_node, req.host_hit_length
    )
    req.prefix_indices = torch.cat([req.prefix_indices, new_indices])
```

`init_load_back` → `load_back` (hiradix_cache.py:872):
1. Walks from evicted node to the nearest GPU-resident ancestor
2. Locks the ancestor (`inc_lock_ref`) to prevent eviction during transfer
3. Concatenates all host indices along the path
4. Calls `cache_controller.load()` → allocates GPU indices + queues async DMA
5. Returns new GPU indices immediately (transfer is async)

### Phase 4: Per-Layer Overlap During Forward Pass

After building the batch:

```python
# scheduler.py:2169
new_batch.hicache_consumer_index = self.tree_cache.ready_to_load_host_cache()
```

`ready_to_load_host_cache()` calls `cache_controller.start_loading()`:

```
start_loading():
  1. Claim a ring buffer slot (producer_id) from LayerDoneCounter (3-slot ring)
  2. Merge all queued load ops into one batched DMA
  3. On the dedicated load_stream:
       for layer_i in 0..N-1:
           load_to_device_per_layer(layer_i)    # copy just this layer
           layer_done_counter.complete(layer_i)  # signal: layer_i ready!
  4. Return producer_id → stored as hicache_consumer_index
```

During the model forward pass, each attention layer calls:

```python
layer_done_counter.wait_until(layer_index)
```

This creates a **producer/consumer pipeline**:

```
Load stream:  [layer0 DMA][layer1 DMA][layer2 DMA][layer3 DMA]...
Compute:             [layer0 attn][layer1 attn][layer2 attn]...
                     ↑ waits for   ↑ waits for
                     layer0 event  layer1 event
```

### Phase 5: Post-Forward → Write-Through and Eviction

After a request finishes:

```
process_batch_result() → release_kv_cache() → tree_cache.cache_finished_req()
  → HiRadixCache.insert()
    → _inc_hit_count(node)
      → if hit_count >= write_through_threshold:
          write_backup(node)       # async GPU → Host on write_stream
            → on completion: write_backup_storage(node)  # async Host → Storage on backup_thread
```

**Eviction** happens when GPU memory is exhausted during allocation:

```python
# HiRadixCache.evict() — hiradix_cache.py:774
for each leaf in eviction_heap (sorted by LRU/LFU/etc):
    if node.backuped:           # host_value exists
        → _evict_backuped(): free GPU indices, set value=None, KEEP node in tree
    elif write_back_policy:
        → write_backup(write_back=True): sync GPU→Host first, then evict
    else:
        → _evict_regular(): free GPU indices, DELETE node from tree
```

**Host eviction** (hiradix_cache.py:839):
- Only evicts nodes already GPU-evicted (`node.evicted`) with `host_ref_counter == 0`
- Frees host indices, deletes node from tree entirely
- Data survives only in L3 storage (if it was written through)

---

## 6. HiCacheController: The Async Transfer Engine

### Architecture

```
[Scheduler Thread]
    write(device_indices)          load(host_indices)
         │                               │
    write_queue                     load_queue
         │                               │
    start_writing()                start_loading() → producer_id
         │                               │
    [write_stream CUDA]            [load_stream CUDA]
    backup_all_layers               load_per_layer[0..N]
    (GPU→Host, all layers)          + LayerDoneCounter events
         │                               │
    ack_write_queue               ack_load_queue
                                         │
                                [Model compute stream]
                                wait_until(layer_i) per layer

[prefetch_thread]                 [backup_thread]
  prefetch_queue                    backup_queue
  _storage_hit_query (TP sync)      _page_backup (host→storage)
  → prefetch_buffer                 ack_backup_queue
  [prefetch_io_aux_thread]
    _page_transfer (storage→host)
    host_mem_release_queue
```

### Key Design Decisions

1. **Merge-on-flush batching**: `start_writing` and `start_loading` drain their entire queue in one merged `CacheOperation`, maximizing DMA bandwidth utilization.
2. **Separate stop events**: `stop_event` (DMA) vs `storage_stop_event` (storage) allows hot attach/detach of storage backends without interrupting GPU↔CPU transfers.
3. **Ring buffer of 3 `LayerLoadingEvent` slots**: prevents producer/consumer aliasing under pipelined execution.
4. **MLA rank-0 backup optimization**: for MLA models, all TP workers share equivalent KV state, so only rank 0 writes to storage (`backup_skip`).

---

## 7. Storage Backend Architecture

### Abstract Interface (hicache_storage.py)

The `HiCacheStorage` base class requires just 3 core operations:

```python
class HiCacheStorage(ABC):
    def get(self, key: str, ...) -> Tensor | None
    def set(self, key: str, value, ...) -> bool
    def exists(self, key: str) -> bool

    # Batch versions
    def batch_get(self, keys, ...) -> List[Tensor | None]
    def batch_set(self, keys, values, ...) -> bool
    def batch_exists(self, keys, ...) -> int  # consecutive existing keys count

    # Zero-copy versions (newer, preferred)
    def batch_get_v1(self, keys, host_indices, ...) -> List[bool]
    def batch_set_v1(self, keys, host_indices, ...) -> List[bool]

    # Lifecycle
    def register_mem_pool_host(self, mem_pool_host: HostKVCache)
    def clear(self) -> None
```

### Content-Addressed Keys

Pages are identified by **SHA256 hashes chained from the prefix** (position-aware):

```
hash(page_0) = SHA256(token_ids_0..63)
hash(page_1) = SHA256(hash(page_0) + token_ids_64..127)
hash(page_N) = SHA256(hash(page_N-1) + token_ids_N*page_size..(N+1)*page_size-1)
```

Same tokens at different positions produce different keys — critical for prefix-dependent KV caches.

### Available Backends

| Backend | Transport | Best For |
|---|---|---|
| `file` | Local filesystem (`readinto`/`tofile`) | Development, single-node |
| `mooncake` | RDMA (zero-copy via transfer engine) | Distributed, high-throughput |
| `hf3fs` | usrbio kernel bypass | HuggingFace 3FS clusters |
| `nixl` | NIXL transfer engine (FILE/OBJ memory) | Plugin-based, flexible |
| `aibrix` | Block-based allocation | AIBrix platform |
| `eic` | RDMA key-value | External inference cache |
| `dynamic` | User-provided module path | Custom integrations |

Backends are lazy-loaded via `StorageBackendFactory` (only imported when first requested).

### Runtime Attach/Detach

Storage backends can be attached/detached at runtime via admin API:

```
POST /attach_hicache_storage   → scheduler.attach_hicache_storage_wrapped()
POST /detach_hicache_storage   → scheduler.detach_hicache_storage_wrapped()
```

Only allowed when the engine is completely idle (no running/queued requests). The `HiRadixCache` host tier continues operating independently.

---

## 8. Write Policies

| Policy | When Backup Happens | Threshold | Trade-off |
|---|---|---|---|
| `write_through` | On every cache insert | `hit_count >= 1` | Highest durability, highest bandwidth |
| `write_through_selective` | Only after second hit | `hit_count >= 2` | Avoids backing up cold entries |
| `write_back` | Only during GPU eviction | N/A (deferred) | Lowest bandwidth, risk of data loss on sudden OOM |

---

## 9. Prefetch Termination Policies

When a request is about to be scheduled but its storage prefetch is still running:

| Policy | Behavior |
|---|---|
| `best_effort` | Terminate immediately, use whatever pages arrived |
| `wait_complete` | Block until all pages are fetched |
| `timeout` | Wait up to a linear deadline proportional to page count |

Configured via `--hicache-storage-prefetch-policy`.

---

## 10. TP (Tensor Parallel) Synchronization

All TP workers must maintain **identical radix tree state**. Since async completions arrive at unpredictable times, HiCache uses a **min-consensus** pattern:

```python
# Example from writing_check():
num_done = len(ack_write_queue)                             # local count
num_done = all_reduce(num_done, ReduceOp.MIN, tp_group)    # global min
for _ in range(num_done):                                    # all workers process same count
    process_completion(ack_write_queue.pop())
```

This ensures every TP worker processes the same number of events, keeping their trees identical without explicit tree replication.

Applied to: `writing_check()`, `loading_check()`, `drain_storage_control_queues()`, and prefetch hit queries.

---

## 11. Eviction Strategies (evict_policy.py)

The eviction heap is a standard `heapq` min-heap. Lower priority values are evicted first.

| Strategy | Priority Key | Semantics |
|---|---|---|
| `LRUStrategy` | `node.last_access_time` | Evict least recently used |
| `LFUStrategy` | `(hit_count, last_access_time)` | Evict least frequently used, LRU tie-break |
| `FIFOStrategy` | `node.creation_time` | Evict oldest created |
| `MRUStrategy` | `-node.last_access_time` | Evict most recently used |
| `FILOStrategy` | `-node.creation_time` | Evict newest created |
| `PriorityStrategy` | `(node.priority, last_access_time)` | Evict lowest priority, LRU tie-break |

Configured via `--radix-eviction-policy` (default: `lru`).

---

## 12. Host Memory Layouts (memory_pool_host.py)

The host KV pool layout affects I/O efficiency for different access patterns:

| Layout | Tensor Shape | Optimized For |
|---|---|---|
| `layer_first` | `[2, layer_num, size, head_num, head_dim]` | Per-layer loading (default) |
| `page_first` | `[2, size, layer_num, head_num, head_dim]` | Zero-copy storage I/O |
| `page_first_direct` | `[2, page_num, layer_num, page_size, head_num, head_dim]` | Direct page-level access |
| `page_head` | `[2, page_num, head_num, page_size, layer_num, head_dim]` | Head-parallel access |

The blog reports `page_first` layout achieves **up to 2× higher throughput** for storage I/O via zero-copy mechanisms.

---

## 13. Request Abort Cleanup

When requests are aborted (scheduler.py:2741):

```python
if self.enable_hicache_storage:
    self.tree_cache.release_aborted_request(req.rid)
```

`release_aborted_request` (hiradix_cache.py:1377):
1. Clears storage hit tracking
2. If there is an ongoing prefetch: terminates it
3. Frees allocated host indices
4. Releases host protection on nodes (`host_ref_counter--`)

---

## 14. End-to-End Summary

```
Request arrives
  │
  ├─ match_prefix() → finds GPU hit + host hit + knows storage might have more
  ├─ prefetch_from_storage() → async storage→host (background threads)
  │
  ▼ (request waits in queue, prefetch runs in background)
  │
Scheduling loop
  ├─ check_hicache_events() → process write/load/storage completions (TP-synced)
  ├─ check_prefetch_progress() → is storage data ready?
  │   └─ If yes: insert into host tree, re-match prefix
  ├─ init_load_back() → async host→GPU (on load_stream)
  ├─ ready_to_load_host_cache() → start per-layer DMA pipeline
  │
  ▼ (batch created with hicache_consumer_index)
  │
Forward pass
  ├─ wait_until(layer_i) → sync per-layer before attention
  │   (compute layer 0 while layer 1 still loading)
  │
  ▼ (generation complete)
  │
Post-forward
  ├─ insert() → add KV to tree
  ├─ _inc_hit_count() → maybe write_backup() (async GPU→Host→Storage)
  │
  ▼ (memory pressure)
  │
Eviction
  ├─ GPU eviction: free GPU indices, keep ghost nodes if backed up
  ├─ Host eviction: free host indices, delete node (data in storage only)
  └─ Storage persists indefinitely until explicit clear
```

The design ensures the **scheduler never blocks on I/O**: prefetch starts at enqueue time, loading overlaps with forward computation per-layer, and write-through happens on background streams/threads.
