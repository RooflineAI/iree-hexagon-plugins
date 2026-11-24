# Hexagon runtime memory model

This document explains how buffers move between the ARM host and the Hexagon
DSP, and why the runtime handles them the way it does. It sits on top of IREE's
HAL buffer and command buffer model and only covers what is specific to
Hexagon.

Code pointers:

- host side: [`mem_alloc.c`](mem_alloc.c), [`buffer.c`](buffer.c),
  [`device.c`](device.c)
- DSP side: [`dsp/residency.h`](dsp/residency.h),
  [`dsp/command_buffer.c`](dsp/command_buffer.c),
  [`dsp/next_use.h`](dsp/next_use.h), [`dsp/buffer.c`](dsp/buffer.c)

All numbers below were measured on one phone (OnePlus 13, SM8750, cDSP v79)
unless marked as an estimate. Other SoCs may differ.

## Platform constraints that drive the design

| Constraint | Consequence |
|---|---|
| The 32-bit DSP address space fits only **3.5-3.75 GiB** of mappings. | Large models (Qwen3-4B: 7.5 GiB) can't be mapped all at once. |
| A mapping always covers a **whole allocation**; sub-ranges and double mappings are rejected. | The allocation is the unit of mapping and eviction ([compiler flags](#what-the-compiler-must-provide)). |
| Registering with `FASTRPC_MAP_FD_DELAYED` uses **no** DSP address space. | The host registers everything once; the DSP maps on demand. |
| The DSP **can't map file-backed memory**; `FASTRPC_ATTR_IMPORT_BUFFER` isn't supported on Android. | Weights are copied into RPCmem once at load. No zero-copy `import_buffer`. |
| `rpcmem_alloc` takes an `int` size. | One allocation is at most `INT_MAX` (~2 GiB). |
| `HAP_mmap2`/`HAP_munmap2` on the DSP: 20-50 us, any size. `fastrpc_mmap` on the host: ~10 ms/GiB. | Mapping per dispatch on the DSP is cheap. |
| With `FASTRPC_MAP_FD_DELAYED`, the user keeps caches coherent. Flush/invalidate: ~17 ms/GiB. | The DSP runtime does all cache maintenance, as little as possible ([cache coherency](#cache-coherency)). |
| Deregistering on the host while the DSP has the allocation mapped **crashes the DSP process** on next access. | The host asks the DSP to unmap before freeing. |
| ~3850 registrations per process, any size; DSP mappings count too (measured, undocumented). | Fine with today's allocation sizes; many small ones could hit it. |

## Buffer lifecycle

1. **Allocate (host).** Every device-visible buffer is RPCmem
   (`rpcmem_alloc`, `RPCMEM_DEFAULT_FLAGS`, i.e. cached and coherent on the ARM
   side). The size is padded so that unaligned or masked HVX vector accesses
   at the end of the buffer stay inside the allocation (see the comment in
   `iree_hal_hexagon_mem_alloc_create`).
2. **Register (host).** `rpcmem_to_fd` + `fastrpc_mmap(fd, FASTRPC_MAP_FD_DELAYED)`.
   This uses no DSP address space, so the total size of all buffers is limited
   by host RAM, not by the DSP address space.
3. **Reference.** Command buffers and binding tables refer to buffers by
   `(fd, offset, length)`. The DSP only receives the fd number; it can only map
   what the host registered. `HAP_GET_FD_INFO` is not supported, so the
   allocation size is sent along with the fd.
4. **Acquire (DSP).** Before a dispatch, copy or fill runs, the DSP acquires
   every buffer it uses from the residency cache. The cache maps the whole
   allocation with `HAP_mmap2` if needed, and keeps it mapped while acquired.
5. **Release (DSP).** After the dispatch, copy or fill finishes, its buffers
   are released. They **stay mapped** until the address space is needed for
   another allocation (see the next section).
6. **Free (host).** When the last reference is dropped,
   `iree_hal_hexagon_mem_alloc_release` first asks the DSP to drop its mapping
   (`hexagon_dsp_buffer_forget`, which flushes and unmaps), then deregisters
   (`fastrpc_munmap`) and frees. If a command on the DSP still uses the buffer,
   the host leaks it rather than crash the DSP. The extra RPC (about 0.3 ms) is
   skipped for buffers that were never handed to the DSP (`dsp_used`).

The fd number of a freed allocation can be reused by a later allocation. That
is safe because the DSP has forgotten the old mapping before the host frees it.

## Residency cache

[`dsp/residency.h`](dsp/residency.h) keeps track of which allocations are
mapped on the DSP.
Keeping buffers mapped until the space is needed makes the common case,
where everything fits, almost free.

### Budget

The cache keeps at most a **budget** of bytes mapped, set with
`--hexagon_dsp_residency_budget_mib` (default
`IREE_HAL_HEXAGON_DEFAULT_DSP_RESIDENCY_BUDGET_MIB` = 3584 MiB in
[`api.h`](api.h)).

- 3584 MiB was chosen to be just below the measured mapping limit. It was
  **not tuned**. The space above it is for the skel and kernel libraries, the
  DSP heap (command buffer copies live there), stacks, FastRPC argument
  mappings and fragmentation.
- The budget is a soft limit. The cache does not know about fragmentation: if
  `HAP_mmap2` fails even though the budget allows it, the cache unmaps more
  and retries.
- The budget must be at least the largest allocation, and at least the sum of
  the bindings of the largest dispatch. If a buffer can never fit, the
  execution fails with `AEE_ENOMEMORY`. `AEE_EWOULDBLOCK` (does not fit right
  now) is handled inside the DSP scheduler and never returned to the host: it
  maps to `IREE_STATUS_DEFERRED`, and the host would retry forever.
- If a buffer does not fit only because the dispatches in flight hold their
  buffers, the scheduler waits for them to finish and tries again.

### What gets unmapped: Belady, then LRU

When a new allocation does not fit, the cache unmaps allocations that are not
acquired, until it fits.

- **With a known future** (command buffer execution), it unmaps the allocation
  whose next use is **furthest in the future** (Belady's rule). Among equals,
  and for allocations without a known next use, it unmaps the least recently
  used one.
- **Without a future**, it unmaps the least recently used one.

Why not plain LRU: model weights are often read in a cycle, layer after layer. When
they don't fit, LRU may unmap exactly the allocation that is needed next.

## Cache coherency

The ARM side of RPCmem is cached and coherent. On the DSP side, the runtime
keeps caches coherent itself. The rules are in
`hexa_cmd_buf_exec_buf` ([`dsp/command_buffer.c`](dsp/command_buffer.c)) and
[`dsp/residency.c`](dsp/residency.c):

- **Execution start:** invalidate every bound range (binding table entries and
  direct references) that is **still mapped** from earlier, because the host
  may have written it since.
- **Between commands:** nothing. All commands of a command buffer run on the
  DSP and share the same view of the caches.
- **Execution end:** flush and invalidate every bound range that is mapped,
  also on error, so the host sees whatever was written.
- **Unmap** (eviction, `forget`, session close): flush and invalidate the
  whole allocation first. After unmapping, no virtual address is left to reach
  those cache lines, and another allocation may be mapped at the same address
  next (freed address space is reused right away).
- **Map:** no invalidate needed. The previous unmap left no lines for these
  addresses.

## Threads and sessions

- **One cache per RPC session.** It is owned by the session's private data on
  the DSP and shared by all command buffers of that session.
- **The address space belongs to the DSP process**, which all sessions of one
  host process share. The budgets of all sessions together must stay below
  what the DSP can map. One session cannot evict another session's unused
  buffers.
- **Only the master thread** (the one running `command_buffer_execute`)
  acquires and releases buffers. Worker threads only run the workgroups of
  dispatches the master has prepared; the master releases their buffers after
  they finish.
- The cache still has a mutex just in case, because other RPCs may arrive on other FastRPC
  threads while an execution runs (`buffer_forget` when the host frees a
  buffer, session close, setting the budget).


## What the compiler must provide

The integration tests compile with these flags
([`integration_tests/stages/compiling.py`](../../../integration_tests/stages/compiling.py)):

| Flag | Value | Why |
|---|---|---|
| `--iree-stream-resource-max-allocation-size` | 1 GiB | Every allocation must stay below `rpcmem_alloc`'s `INT_MAX` limit and be mappable as a whole within the budget. Smaller allocations also let eviction work at a finer grain. The value was **not tuned**. |
| `--iree-dispatch-creation-max-dispatch-binding-size` | 3 GiB | All bindings of one dispatch are mapped at the same time, so they must fit into the budget together. 3 GiB stays below the default budget of 3.5 GiB. |

Programs compiled without these flags may still work, as long as their allocations and dispatches fit into the budget.
