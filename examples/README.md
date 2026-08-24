# Examples

Two small models that exercise the Hexagon compiler and runtime plugins end to end.

| Example | Entry point | Inputs | Notes |
|---|---|---|---|
| [`matmul/matmul.mlir`](matmul/matmul.mlir) | `matmul` | 2 × `32x32xf16` | Single `linalg.matmul`, f16 in / f32 out. Start here. |
| [`attention_layer/attention_layer.mlir`](attention_layer/attention_layer.mlir) | `main` | 3 × `4x1024x128xf16` | Torch-imported attention block; exercises the full lowering pipeline (without fast attention). |

Both are provided here in plain MLIR, so there is no import or download step.

These examples have been tested on a OnePlus 13 phone with Snapdragon 8 Elite SoC, model number SM7850, containing a Hexagon DSP v79.

## Two paths

**Compiling needs no Hexagon hardware.** You can build `iree-compile`, produce a `.vmfb`
for the DSP, and read the lowering log on any Linux x86_64 machine. That is
[compiling.md](compiling.md), and it is where to start.

**Executing on the DSP needs a phone** with a Hexagon DSP, connected over ADB. That is
[running.md](running.md).

## Prerequisites

Before either document, follow the steps from the top-level [README](../README.md) and make sure that you are able to build the compiler.

> The first build compiles LLVM, MLIR and IREE from source. Expect it to take a long time and
> tens of GB of disk on a cold cache.
