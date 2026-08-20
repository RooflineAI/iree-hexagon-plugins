// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef ROOF_HEXAGON_CODEGEN_IR_HEXAGONENUMS
#define ROOF_HEXAGON_CODEGEN_IR_HEXAGONENUMS

include "hexagon/CodeGen/IR/HexagonDialect.td"
include "mlir/IR/EnumAttr.td"

class IREEHexagon_I32EnumAttr<string name, string summary,
                              list<I32EnumAttrCase> cases>
    : I32EnumAttr<name, summary, cases> {
  let cppNamespace = "::mlir::iree_compiler::IREE::Hexagon";
  // The enum is wrapped in a dialect attribute (see HexagonAttrs.td), so no
  // specialized IntegerAttr-based attribute is generated here.
  let genSpecializedAttr = 0;
}

//===----------------------------------------------------------------------===//
// Profiler zone types.
//===----------------------------------------------------------------------===//

// The cases mirror HEXAGON_PROFILER_ZONES in
// plugins/runtime/hexagon/arm_dsp/profiler.h, in declaration order. The values
// are passed to the DSP runtime unchanged and index the zone name table there,
// so the two lists must be kept in sync.
def IREEHexagon_ProfilerZoneDspExecution :
    I32EnumAttrCase<"DspExecution", 0, "dsp_execution">;
def IREEHexagon_ProfilerZoneDispatch :
    I32EnumAttrCase<"Dispatch", 1, "dispatch">;
def IREEHexagon_ProfilerZoneKernel :
    I32EnumAttrCase<"Kernel", 2, "kernel">;
def IREEHexagon_ProfilerZoneBarrier :
    I32EnumAttrCase<"Barrier", 3, "barrier">;
def IREEHexagon_ProfilerZoneCopy :
    I32EnumAttrCase<"Copy", 4, "copy">;
def IREEHexagon_ProfilerZoneFill :
    I32EnumAttrCase<"Fill", 5, "fill">;
def IREEHexagon_ProfilerZoneMemoryManagement :
    I32EnumAttrCase<"MemoryManagement", 6, "memory_management">;
def IREEHexagon_ProfilerZoneMarker :
    I32EnumAttrCase<"Marker", 7, "marker">;
def IREEHexagon_ProfilerZoneUnknown :
    I32EnumAttrCase<"Unknown", 8, "unknown">;

def IREEHexagon_ProfilerZone :
    IREEHexagon_I32EnumAttr<"ProfilerZone",
                            "Type of a Hexagon DSP profiler zone", [
      IREEHexagon_ProfilerZoneDspExecution,
      IREEHexagon_ProfilerZoneDispatch,
      IREEHexagon_ProfilerZoneKernel,
      IREEHexagon_ProfilerZoneBarrier,
      IREEHexagon_ProfilerZoneCopy,
      IREEHexagon_ProfilerZoneFill,
      IREEHexagon_ProfilerZoneMemoryManagement,
      IREEHexagon_ProfilerZoneMarker,
      IREEHexagon_ProfilerZoneUnknown
    ]>;

#endif // ROOF_HEXAGON_CODEGEN_IR_HEXAGONENUMS
