# LLVM patches

Applied to the `third-party/llvm-project` working tree by
[`apply_submodule_patches.sh`](../../build_tools/apply_submodule_patches.sh).

| Patch | Category | Reason |
| --- | --- | --- |
| [hexagon_reserve_v30_backport_pr243763.patch](hexagon_reserve_v30_backport_pr243763.patch) | Hexagon backend miscompile, upstream backport | Unmodified backport of llvm/llvm-project#243763 (merged 2026-11-23, `cf97459403f3`), which fixes llvm/llvm-project#237215. `HexagonPostRAHandleQFP` materializes a zero vector in a hard-coded `V30` when it inserts IEEE->qf conversions after register allocation, without checking liveness; nothing reserved V30, so the pass overwrote live values. On v79 the patch reserves V30 in `getBaseReservedRegs` (non-legacy qfloat modes always; legacy mode when the function has QFP instructions). Fixes the Qwen3-VL-4B vision encoder (hexagon-vtcm-hmx: 67-90% -> 0.59-0.71% rel-L2) and the Qwen3-VL-2B vision encoder in hexagon-vtcm-hmx (0.85% rel-L2). Adds `autohvx/xqf-reserve-v30.ll`. **Drop at the next llvm-project bump past `cf97459403f3`.** |
| [hexagon_postra_qfp_partial_pair_refill.patch](hexagon_postra_qfp_partial_pair_refill.patch) | Hexagon backend miscompile | ROO-2035, re-applied: it was dropped at the LLVM upgrade although upstream does not have it. `HexagonPostRAHandleQFP::collectConvQFInstr` treated a register-pair stack refill feeding `V6_vconv_hf_qf32` as defining both halves, and re-converted a half that was already qf32. The fix converts only the halves the refill still reaches. Fixes the Qwen3-VL-2B vision encoder in the plain `hexagon` case (NaN -> 0.85% rel-L2). Adds `autohvx/xqf-postra-conv-double-partial-refill.mir`. Details: `testing/agent_docs/investigations/qwen3vl-vision-probs-v-qfp-refill.md` (local). |
