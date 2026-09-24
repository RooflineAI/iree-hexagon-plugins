// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hexagon/CodeGen/Bufferization/HexagonBufferizableOpInterfaceImpl.h"

#include "hexagon/CodeGen/IR/HexagonDialect.h"
#include "hexagon/CodeGen/IR/HexagonOps.h"

#include "mlir/Dialect/Bufferization/IR/DstBufferizableOpInterfaceImpl.h"

using namespace mlir;
using namespace mlir::bufferization;

namespace mlir::iree_compiler::IREE::Hexagon {

namespace {

// Generic bufferization for tensor-level HMX ops. These ops are pure tensor
// destination-passing-style operations; bufferization materializes the matching
// side-effecting buffer op and replaces the tensor result with the init buffer.
template <typename TensorOpTy, typename BufferOpTy>
struct HmxDpsOpInterface
    : public DstBufferizableOpInterfaceExternalModel<
          HmxDpsOpInterface<TensorOpTy, BufferOpTy>, TensorOpTy> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    auto dpsOp = cast<DestinationStyleOpInterface>(op);
    if (!dpsOp.isDpsInit(&opOperand)) {
      return true;
    }
    // Pack and matmul fully overwrite their destinations. Unpack adds the HMX
    // product to the original matmul initializer and therefore reads it.
    return isa<TensorHmxUnpackOp>(op);
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options,
                          BufferizationState &state) const {
    auto dpsOp = cast<DestinationStyleOpInterface>(op);
    // Already bufferized.
    if (dpsOp.hasPureBufferSemantics()) {
      return success();
    }
    if (!dpsOp.hasPureTensorSemantics()) {
      return op->emitError() << "op does not have pure tensor semantics";
    }

    SmallVector<Value> bufferOperands;
    bufferOperands.reserve(op->getNumOperands());
    for (Value operand : op->getOperands()) {
      if (isa<TensorType>(operand.getType())) {
        FailureOr<Value> buffer = getBuffer(rewriter, operand, options, state);
        if (failed(buffer)) {
          return failure();
        }
        bufferOperands.push_back(*buffer);
      } else {
        bufferOperands.push_back(operand);
      }
    }

    // The init is the last operand; the tensor result aliases its buffer.
    Value initBuffer = bufferOperands.back();

    OperationState newState(op->getLoc(), BufferOpTy::getOperationName());
    newState.addOperands(bufferOperands);
    newState.addAttributes(op->getAttrs());
    rewriter.create(newState);

    replaceOpWithBufferizedValues(rewriter, op, initBuffer);
    return success();
  }
};

} // namespace

void registerBufferizableOpInterfaceExternalModels(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, IREEHexagonDialect *dialect) {
    TensorHmxMatmulOp::attachInterface<
        HmxDpsOpInterface<TensorHmxMatmulOp, HmxMatmulOp>>(*ctx);
    TensorHmxPackOp::attachInterface<
        HmxDpsOpInterface<TensorHmxPackOp, HmxPackOp>>(*ctx);
    TensorHmxUnpackOp::attachInterface<
        HmxDpsOpInterface<TensorHmxUnpackOp, HmxUnpackOp>>(*ctx);
  });
}

} // namespace mlir::iree_compiler::IREE::Hexagon
