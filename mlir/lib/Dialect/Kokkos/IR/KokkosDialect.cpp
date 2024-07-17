// ===- PartTensorDialect.cpp - part_tensor dialect implementation -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Kokkos/IR/KokkosDialect.h"
#include <utility>

#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

#include "mlir/Dialect/Kokkos/IR/KokkosEnums.cpp.inc"

//#define GET_ATTRDEF_CLASSES
//#include "mlir/Dialect/Kokkos/IR/KokkosAttrDefs.cpp.inc"

using namespace mlir;
using namespace mlir::kokkos;

void KokkosDialect::initialize() {
//  addAttributes<
//#define GET_ATTRDEF_LIST
//#include "mlir/Dialect/Kokkos/IR/KokkosAttrDefs.cpp.inc"
//      >();
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/Kokkos/IR/Kokkos.cpp.inc"
      >();
}

void ParallelOp::build(
    OpBuilder &builder, OperationState &result, ::mlir::kokkos::ExecutionSpace executionSpace, ::mlir::kokkos::ParallelLevel parallelLevel,
    ValueRange upperBounds, ValueRange initVals,
    function_ref<void(OpBuilder &, Location, ValueRange, ValueRange)>
        bodyBuilderFn) {
  result.addOperands(upperBounds);
  result.addOperands(initVals);
  result.addAttribute(
      ParallelOp::getOperandSegmentSizeAttr(),
      builder.getDenseI32ArrayAttr({static_cast<int32_t>(upperBounds.size()),
                                    static_cast<int32_t>(initVals.size())}));
  result.addAttribute("executionSpace", ExecutionSpaceAttr::get(builder.getContext(), executionSpace));
  result.addAttribute("parallelLevel", ParallelLevelAttr::get(builder.getContext(), parallelLevel));
  result.addTypes(initVals.getTypes());

  OpBuilder::InsertionGuard guard(builder);
  unsigned numIVs = upperBounds.size();
  SmallVector<Type, 8> argTypes(numIVs, builder.getIndexType());
  SmallVector<Location, 8> argLocs(numIVs, result.location);
  Region *bodyRegion = result.addRegion();
  Block *bodyBlock = builder.createBlock(bodyRegion, {}, argTypes, argLocs);

  if (bodyBuilderFn) {
    builder.setInsertionPointToStart(bodyBlock);
    bodyBuilderFn(builder, result.location,
                  bodyBlock->getArguments().take_front(numIVs),
                  bodyBlock->getArguments().drop_front(numIVs));
  }
  ParallelOp::ensureTerminator(*bodyRegion, builder, result.location);
}

void ParallelOp::build(
    OpBuilder &builder, OperationState &result, ::mlir::kokkos::ExecutionSpace executionSpace, ::mlir::kokkos::ParallelLevel parallelLevel,
    ValueRange upperBounds, function_ref<void(OpBuilder &, Location, ValueRange)> bodyBuilderFn) {
  // Only pass a non-null wrapper if bodyBuilderFn is non-null itself. Make sure
  // we don't capture a reference to a temporary by constructing the lambda at
  // function level.
  auto wrappedBuilderFn = [&bodyBuilderFn](OpBuilder &nestedBuilder,
                                           Location nestedLoc, ValueRange ivs,
                                           ValueRange) {
    bodyBuilderFn(nestedBuilder, nestedLoc, ivs);
  };
  function_ref<void(OpBuilder &, Location, ValueRange, ValueRange)> wrapper;
  if (bodyBuilderFn)
    wrapper = wrappedBuilderFn;

  build(builder, result, executionSpace, parallelLevel, upperBounds, ValueRange(), wrapper);
}

Region &ParallelOp::getLoopBody() { return getRegion(); }

ParseResult ParallelOp::parse(OpAsmParser &parser, OperationState &result) {
  auto &builder = parser.getBuilder();
  // Parse an opening `(` followed by induction variables followed by `)`
  SmallVector<OpAsmParser::Argument, 4> ivs;
  if (parser.parseArgumentList(ivs, OpAsmParser::Delimiter::Paren))
    return failure();

  // Parse loop bounds.
  SmallVector<OpAsmParser::UnresolvedOperand, 4> upper;
  if (parser.parseArrow() ||
      parser.parseOperandList(upper, ivs.size(),
                              OpAsmParser::Delimiter::Paren) ||
      parser.resolveOperands(upper, builder.getIndexType(), result.operands))
    return failure();

  // Parse init values.
  SmallVector<OpAsmParser::UnresolvedOperand, 4> initVals;
  if (succeeded(parser.parseOptionalKeyword("init"))) {
    if (parser.parseOperandList(initVals, OpAsmParser::Delimiter::Paren))
      return failure();
  }

  // Parse optional results in case there is a reduce.
  if (parser.parseOptionalArrowTypeList(result.types))
    return failure();

  // Now parse the body.
  Region *body = result.addRegion();
  for (auto &iv : ivs)
    iv.type = builder.getIndexType();
  if (parser.parseRegion(*body, ivs))
    return failure();

  // Set `operandSegmentSizes` attribute.
  result.addAttribute(
      ParallelOp::getOperandSegmentSizeAttr(),
      builder.getDenseI32ArrayAttr({static_cast<int32_t>(upper.size()),
                                    static_cast<int32_t>(initVals.size())}));

  // Parse attributes.
  if (parser.parseOptionalAttrDict(result.attributes) ||
      parser.resolveOperands(initVals, result.types, parser.getNameLoc(),
                             result.operands))
    return failure();

  // Add a terminator if none was parsed.
  mlir::scf::ForOp::ensureTerminator(*body, builder, result.location);
  return success();
}

void ParallelOp::print(OpAsmPrinter &p) {
  p << " (" << getBody()->getArguments() << ") -> (" << getUpperBound() << ")";
  if (!getInitVals().empty())
    p << " init (" << getInitVals() << ")";
  p.printOptionalArrowTypeList(getResultTypes());
  p << ' ';
  p.printRegion(getRegion(), /*printEntryBlockArgs=*/false);
  p.printOptionalAttrDict(
      (*this)->getAttrs(),
      /*elidedAttrs=*/ParallelOp::getOperandSegmentSizeAttr());
}

void ParallelOp::getSuccessorRegions(
    RegionBranchPoint point, SmallVectorImpl<RegionSuccessor> &regions) {
  // Both the operation itself and the region may be branching into the body or
  // back into the operation itself. It is possible for loop not to enter the
  // body.
  regions.push_back(RegionSuccessor(&getRegion()));
  regions.push_back(RegionSuccessor());
}

template <typename TerminatorTy>
static TerminatorTy verifyAndGetTerminator(Operation *op, Region &region,
                                           StringRef errorMessage) {
  Operation *terminatorOperation = nullptr;
  if (!region.empty() && !region.front().empty()) {
    terminatorOperation = &region.front().back();
    if (auto yield = dyn_cast_or_null<TerminatorTy>(terminatorOperation))
      return yield;
  }
  auto diag = op->emitOpError(errorMessage);
  if (terminatorOperation)
    diag.attachNote(terminatorOperation->getLoc()) << "terminator here";
  return nullptr;
}

LogicalResult ParallelOp::verify() {
  // Check that there is at least one value in upperBound.
  if (getUpperBound().empty())
    return emitOpError(
        "needs at least one tuple element for upperBound");
  auto loopDim = getUpperBound().size();

  // Check that the body defines the same number of block arguments as there
  // are upper bounds.
  Block *body = getBody();
  if (body->getNumArguments() != loopDim)
    return emitOpError() << "expects the same number of induction variables: "
                         << body->getNumArguments()
                         << " as bounds: " << loopDim;
  for (auto arg : body->getArguments())
    if (!arg.getType().isIndex())
      return emitOpError(
          "expects arguments for the induction variable to be of index type");

  // Check that the yield has no results
  auto yield = verifyAndGetTerminator<scf::YieldOp>(
      *this, getRegion(), "expects body to terminate with 'scf.yield'");
  if (!yield)
    return failure();
  if (yield->getNumOperands() != 0)
    return yield.emitOpError() << "not allowed to have operands inside '"
                               << ParallelOp::getOperationName() << "'";

  // Check that the number of results is the same as the number of ReduceOps.
  SmallVector<mlir::scf::ReduceOp, 4> reductions(body->getOps<mlir::scf::ReduceOp>());
  auto resultsSize = getResults().size();
  auto reductionsSize = reductions.size();
  auto initValsSize = getInitVals().size();
  if (resultsSize != reductionsSize)
    return emitOpError() << "expects number of results: " << resultsSize
                         << " to be the same as number of reductions: "
                         << reductionsSize;
  if (resultsSize != initValsSize)
    return emitOpError() << "expects number of results: " << resultsSize
                         << " to be the same as number of initial values: "
                         << initValsSize;

  // Check that the types of the results and reductions are the same.
  for (auto resultAndReduce : llvm::zip(getResults(), reductions)) {
    auto resultType = std::get<0>(resultAndReduce).getType();
    auto reduceOp = std::get<1>(resultAndReduce);
    auto reduceType = reduceOp.getOperand().getType();
    if (resultType != reduceType)
      return reduceOp.emitOpError()
             << "expects type of reduce: " << reduceType
             << " to be the same as result type: " << resultType;
  }
  return success();
}

#define GET_OP_CLASSES
#include "mlir/Dialect/Kokkos/IR/Kokkos.cpp.inc"

#include "mlir/Dialect/Kokkos/IR/KokkosDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// convenience methods.
//===----------------------------------------------------------------------===//

