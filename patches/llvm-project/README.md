# LLVM patches

Local LLVM fixes applied to `third-party/llvm-project` before building. Each
patch includes a regression under `llvm/test/` for independent upstream review.

| Patch | Category | Reason |
| --- | --- | --- |
| [0001-hexagon_postra_qfp_partial_pair_refill.patch](0001-hexagon_postra_qfp_partial_pair_refill.patch) | Hexagon backend correctness | Convert only the IEEE halves of a refilled register pair that still reach a qfloat conversion. |
| [0002-widen_setcc_inputs_wider_than_result.patch](0002-widen_setcc_inputs_wider_than_result.patch) | SelectionDAG type legalization | Compare at the widened operands' lane count, then extract the required result lanes. |

## Partially refilled qfloat pairs

`HexagonPostRAHandleQFP` converts refilled IEEE values to qfloat before a
pair-consuming conversion. If one half of the pair has since been redefined by
a qfloat producer, converting it again interprets qfloat bits as IEEE and
corrupts the result.

The patch scans backward in the same block to identify which halves still hold
the refill and converts only those halves. The RDF graph represents the pair
use as reaching the refill and does not distinguish these intervening half
redefinitions. Cases without a same-block definition retain the existing
behavior; the dataflow assumptions deserve review before upstream submission.
The MIR regression covers low-half and high-half redefinitions and an unchanged
pair.

## Compare operands wider than the result

A vector comparison's operands can widen beyond its predicate result. For
example, on Hexagon, 25 f16 lanes widen to 64 operand lanes while the predicate
result widens to 32 lanes. A comparison must have matching operand and result
lane counts.

The patch compares at the operands' widened width and extracts the leading
result lanes. The extra lanes correspond to padding and do not change the
retained comparisons. The regression covers 17-, 25- and 31-lane f16
comparisons. This fixes type legalization; scalarized division and excessive
stack usage in fused erf-GELU are separate problems.

## Running the regressions

Apply the patches, then build the LLVM tools with Bazel:

```sh
build_tools/apply_submodule_patches.sh
bazel build @llvm-project//llvm:llc @llvm-project//llvm:FileCheck
llvm_llc="$(bazel cquery --output=files @llvm-project//llvm:llc)"
llvm_filecheck="$(bazel cquery --output=files @llvm-project//llvm:FileCheck)"
llvm_tests=third-party/llvm-project/llvm/test/CodeGen/Hexagon/autohvx
```

The following commands execute the lit tests' `RUN` lines directly, using this
checkout's patched tools. `pipefail` ensures a compiler failure is reported:

```sh
set -o pipefail
"$llvm_llc" -march=hexagon -mcpu=hexagonv79 \
  -mattr=+hvxv79,+hvx-length128B \
  -run-pass=handle-qfp-spills-refills -verify-machineinstrs \
  "$llvm_tests/xqf-postra-conv-double-partial-refill.mir" -o - \
  | "$llvm_filecheck" "$llvm_tests/xqf-postra-conv-double-partial-refill.mir"

"$llvm_llc" -mtriple=hexagon -mcpu=hexagonv79 \
  -mattr=+hvxv79,+hvx-length128b \
  < "$llvm_tests/widen-setcc-fp-non-pow2.ll" \
  | "$llvm_filecheck" "$llvm_tests/widen-setcc-fp-non-pow2.ll"
```

A complete LLVM lit suite additionally needs an LLVM build configured with its
tests enabled; the commands above check just these two patch regressions.
