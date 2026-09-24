# Hexagon dispatch planner

This directory contains the implementation behind `initHexagonLaunchConfig`.
It analyzes a dispatch, selects a lowering strategy, prepares MLIR attributes,
and applies them after the plan has been verified.

The public facade is [`KernelDispatch.cpp`](../KernelDispatch.cpp).

## Planning flow

[`DispatchPlanner.cpp`](DispatchPlanner.cpp) is the orchestrator.
The expected flow is:

| Phase | Input | Output | Scope | Implementation |
| --- | --- | --- | --- | --- |
| Preflight | Entry point and `PlanningOptions` | `PlanningContext`, or an early default/no-op result | Dispatch | [`DispatchPlanner.cpp`](DispatchPlanner.cpp) |
| 1. Analyze dispatch | Compute operations | `DispatchShape` containing the root and local/global dimension relationships | Dispatch | [`DispatchAnalysis.cpp`](DispatchAnalysis.cpp) |
| 2. Select strategy | Dispatch shape and target/options | `DispatchStrategy` with pipeline choice and root-owned tiling | Dispatch/root | [`StrategySelection.cpp`](StrategySelection.cpp), [`Strategies/`](Strategies/) |
| 3. Resolve pipeline contract | Selected pipeline | `PipelineContract` describing downstream pipeline semantics | Pipeline | [`PipelineContract.cpp`](PipelineContract.cpp) |
| 4. Plan VTCM | Strategy, dispatch shape, contract, and options | Finalized strategy with an optional root `VTCMPlan` and reconciled cache tiles | Dispatch/root | [`VTCMPlanning.cpp`](VTCMPlanning.cpp) |
| 5. Select compute tiles | Finalized strategy, contract, and one non-root `OpShape` | Optional independent `OpComputeTilePlan` | Per non-root operation | [`ComputeTileSelection.cpp`](ComputeTileSelection.cpp) |
| 6. Propagate root tile bounds | Complete `DispatchPlan` and contract | Non-root tiles bounded by the root's fusion tile | Dispatch | [`RootTilePropagation.cpp`](RootTilePropagation.cpp) |
| 7. Verify | Complete `DispatchPlan` and contract | Success or a diagnostic | Dispatch | [`PlanVerification.cpp`](PlanVerification.cpp) |
| 8. Encode | Verified plan | Typed `EncodedDispatchPlan` attributes | Dispatch/per operation | [`PlanEncoding.cpp`](PlanEncoding.cpp) |
| 9. Apply | Encoded plan | Mutated IR | Dispatch | [`PlanEncoding.cpp`](PlanEncoding.cpp) |

Phases 1–8 are read-only with respect to the IR. Applying the encoded plan is
the only planner step allowed to mutate it.

## Main contracts

The shared data structures live in
[`DispatchPlanTypes.h`](DispatchPlanTypes.h):

- `DispatchShape` records the selected root, stable operation ordinals, and
  dimension relationships.
- `DispatchStrategy` owns the pipeline choice and `RootTilingPlan`.
- `OpComputeTilePlan` contains one tile decision per local dimension of a non-root
  operation.
- `DispatchPlan` combines the root strategy with the selected non-root plans.

`PipelineContract` is the boundary between planning and lowering. It records
expectations from the different lowering pipelines for the output from strategy selection.
Together with the verification phase, they consist of a way of making the implicit
contract between the heuristic selection and the lowering pipelines explicit.
Nevertheless, they must be manually kept in sync and might require frequent updates.
This is a compromise aimed at making the logic simpler to understand.

The central ownership rules are:

1. Distribution, cache, root compute tiling, and VTCM belong only to
   `RootTilingPlan`.
2. Non-root compute tiles are selected independently from each operation's own
   shape, then bounded by the root's fusion tile where the two overlap.
   Unlike in LLVMCPU, the root's plan is never rewritten by a non-root preference.
3. A tile becomes Common, Reduction, or Inner only during encoding, using its
   iterator type and whether its global dimension is present in the root.
4. Verification and encoding must complete before any attribute is attached.

## Strategy selection

Strategies are tried in the order declared by
[`getStrategyCandidates`](StrategySelection.cpp).

Each candidate returns one of three results:

- a strategy when it matches;
- `nullopt` when the next candidate should be tried;
- diagnosed failure when it matched but could not construct a valid plan.

The registry contains only specialized candidates. If the root has no
analyzable shape or no candidate matches, the driver selects CPUDefault as a
total fallback.

Each operation-family file owns both its root strategy and its non-root
compute-tile policy. Ordinary, HMX, and unsupported contractions share
[`Strategies/ContractionStrategy.cpp`](Strategies/ContractionStrategy.cpp).
Reusable cross-family primitives are in
[`Strategies/StrategySupport.cpp`](Strategies/StrategySupport.cpp).

*Expected future direction*:
There is currently no separate operation-analysis phase. It can be introduced
when a strategy consumes a concrete reusable result, such as an estimated
register footprint per dimension or the expected vectorized dimension.
This was skipped during the initial implementation, but the expectation
is to implement some sort of greedy algorithm that uses a query on each op
to estimate the number of expected registers to be held live.
It should then combine this information together with the expected live ops at
any given moment in the dispatch in order to make better tiling decisions and
avoid register spills or tiles growing excessively and slowing down (or crashing) compilation.

## VTCM planning

VTCM planning runs after strategy selection and pipeline-contract resolution,
but before non-root compute tiles are selected. The strategy has already
chosen its ordinary root distribution, cache, and compute tiles at this point.
[`planVTCMTiling`](VTCMPlanning.cpp) may add a root-only `VTCMPlan` and adjust
the root cache tile according to the pipeline contract; it never mutates IR.

The pipeline contract determines how VTCM availability is handled:

- `Unsupported` pipelines skip VTCM planning.
- `Optional` pipelines attempt VTCM planning when enabled, but retain their
  ordinary strategy when VTCM is disabled, the root is not a tensor-semantics
  Linalg operation, or the Hexagon tiling utility cannot derive tile sizes.
- `Required` pipelines diagnose the same conditions instead of silently
  falling back. HMX uses this contract because its lowering requires VTCM.

On success, `hexagon::determineTileSizes` remains responsible for VTCM capacity
and layout policy. The planner records those sizes on the root and then either
preserves or suppresses ordinary cache tiling according to
`CacheTilingWithVTCM`. Suppression is rejected if it would discard a
hardware-fixed cache tile. Optional failures and successful cache adjustments
are recorded by `DecisionTrace` under the resource stage.

## Root tile propagation (work in progress)

[`RootTilePropagation.cpp`](RootTilePropagation.cpp) bounds every
non-root tile by the root's, for pipelines that anchor
Common tiling on the root — the case where a fused operation computes the root's
tile rather than its own. `RootFusionBounds` decides which root tile applies: a
parallel tile bounds every fused operation, a reduction tile bounds only the
producers of the root's input operands, and a dimension absent from the root is
not bounded (that is what `VectorInnerParallel` expresses). An operation a
later pass can still refine keeps the largest divisor of that bound; one that
cannot is matched to it exactly. Verification re-checks all three conditions
independently, in order to avoid implicit assumptions from the lowering pipelines
being broken.

Influence only ever flows away from the root:
unlike LLVMCPU, a non-root preference never rewrites the root's plan,
and conflicts are resolved by shrinking the non-root operation.
This version is only an initial implementation that will be
revisited in the future.

## Decision tracing

[`DecisionTrace`](DecisionTrace.h) records analysis, selection, rejection,
adjustment, verification, and encoding decisions using stable operation
ordinals. In a `dbg` or `opt_dbg` build, enable it with:

```text
--debug-only=iree-hexagon-dispatch-planning
```

The trace uses LLVM debug logging and is therefore unavailable in `opt` builds.

Currently, decisions are simple enough, but the logic is expected to grow enough
to make this difficult to handle. Therefore, this decision tracing is currently
introduced with the objective of making this pass more understandable in the future.

## Other future improvements (TODO)

In its current state, Hexagon Strategy selection is aiming at having different tile sizes on the same tiling-level.
This is currently being translated into two different nested loops through the LLVMCPU's tiling passes.
This is required because different hardware units will dictate different tile sizes and it may therefore not be possible to tile them all together.
Nevertheless, a cleaner approach to this would represent this through different tiling levels.
That would requires considerable work of modularizing/rewriting the LLVMCPU tiling passes and is open work.
Another example of this is the HMX pipeline using cache tiling along with VTCM.
This is a completely arbitrary application of a tiling level of LLVMCPU to Hexagon for a completely different purpose and should also be given its own level.

In the same direction, we are currently using a dual lowering configuration with attributes from LLVMCPU and custom ones for Hexagon.
Moving this logic into the custom one for Hexagon require the same work as the issue above.

This is currently a huge pass combining a considerable amount of logic from multiple files and creating intermediate structures parallel to the IR.
This contradicts MLIR conventions and could be reworked in the future to be split into multiple passes, writing temporary stuff to the IR so that it may more easily be checked and unit tested.
