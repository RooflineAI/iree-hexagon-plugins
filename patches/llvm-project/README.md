# LLVM patches

Applied to the `third-party/llvm-project` working tree by
[`apply_submodule_patches.sh`](../../build_tools/apply_submodule_patches.sh).

| Patch | Category | Reason |
| --- | --- | --- |
| [hexagon_packetizer_slot0_store_order.patch](hexagon_packetizer_slot0_store_order.patch) | Hexagon backend miscompile | Not upstream (checked against llvm-project main on 2026-11-23). When both stores of a packet write the same location, the slot-0 store takes effect last, and the shuffler keeps program order by putting the first store in slot 1. An HVX `vmem` store can only go in slot 0, so a `vmem` store followed by an aliasing scalar store in one packet commits in the wrong order. `HexagonPacketizerList::isLegalToPacketizeTogether` allowed that pair; it already refused the analogous load pair. A dynamic `insertelement` into a vector on the stack (spill, element store, reload) lost every element, so a strided copy such as Qwen3-VL's interleaved M-RoPE (`x[..., 1:60:3] = y[..., 1:60:3]`) wrote zeros. The patch refuses to packetize a store with a later, possibly aliasing store when the first must go in slot 0, except `ALLOCFRAME`, which is glued to the following store on purpose. The Hexagon CodeGen lit tests runnable with `llc`/`FileCheck` (1576) fail identically with and without it, apart from the new test. Adds `autohvx/hvx-store-packet-order.mir`. **Upstream it, then drop it at the bump that contains it.** |
