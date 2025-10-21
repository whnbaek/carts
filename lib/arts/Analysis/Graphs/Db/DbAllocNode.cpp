///==========================================================================
/// DbAllocNode.cpp - Implementation of DbAlloc node
///==========================================================================

#include "arts/Analysis/Db/DbAnalysis.h"
#include "arts/Analysis/Graphs/Db/DbNode.h"
#include "arts/Utils/ArtsUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinTypes.h"
#include <algorithm>
#include <limits>

using namespace mlir;
using namespace mlir::arts;

#include "arts/Utils/ArtsDebug.h"
ARTS_DEBUG_SETUP(db_alloc_node);

//===----------------------------------------------------------------------===//
// DbAllocNode
//===----------------------------------------------------------------------===//
DbAllocNode::DbAllocNode(DbAllocOp op, DbAnalysis *analysis)
    : dbAllocOp(op), dbFreeOp(nullptr), op(op.getOperation()),
      analysis(analysis) {
  ARTS_DEBUG(" - Creating DbAllocNode for operation: " << op);
  info.isAlloc = true;
  info.estimatedBytes = 0;
  info.staticBytes = 0;
  info.sizes.reserve(op.getSizes().size());
  info.sizes.assign(op.getSizes().begin(), op.getSizes().end());

  /// Find the corresponding DbFreeOp for this allocation
  Value dbVal = op.getPtr();
  for (Operation *user : dbVal.getUsers()) {
    if (auto freeOp = dyn_cast<DbFreeOp>(user)) {
      dbFreeOp = freeOp;
      break;
    }
  }

  /// Compute allocation size in bytes from payload sizes if available.
  /// Payload sizes represent the dimensions of the allocated data structure.
  unsigned long long elemBytes =
      arts::getElementTypeByteSize(op.getElementType());

  if (!op.getElementSizes().empty()) {
    bool allConst = true;
    unsigned long long payloadElems = 1;

    /// Compute total element count from payload dimensions
    for (Value v : op.getElementSizes()) {
      if (auto c = v.getDefiningOp<arith::ConstantIndexOp>()) {
        unsigned long long mul =
            (unsigned long long)std::max<int64_t>(c.value(), 1);
        /// Check for overflow
        if (payloadElems >
            std::numeric_limits<unsigned long long>::max() / mul) {
          payloadElems = std::numeric_limits<unsigned long long>::max();
          /// Still treat as const but saturated
          allConst = true;
          break;
        }
        payloadElems *= mul;
      } else {
        allConst = false;
        break;
      }
    }

    /// If all dimensions are constant, compute static byte size
    if (elemBytes > 0 && allConst) {
      unsigned long long bytes = 0;
      if (payloadElems > 0) {
        if (payloadElems >
            std::numeric_limits<unsigned long long>::max() / elemBytes)
          bytes = std::numeric_limits<unsigned long long>::max();
        else
          bytes = payloadElems * elemBytes;
      }
      info.staticBytes = bytes;
      info.estimatedBytes = bytes;
    }
  }
}

void DbAllocNode::print(llvm::raw_ostream &os) const {
  os << "DbAllocNode (" << getHierId() << ")";
  os << " staticBytes=" << info.staticBytes;
  os << "\n";
}

void DbAllocNode::forEachChildNode(
    const std::function<void(NodeBase *)> &fn) const {
  for (const auto &n : acquireNodes)
    fn(n.get());
}

DbAcquireNode *DbAllocNode::getOrCreateAcquireNode(DbAcquireOp op) {
  auto it = acquireMap.find(op);
  if (it != acquireMap.end())
    return it->second;
  std::string childId =
      (getHierId() + "." + std::to_string(nextChildId++)).str();
  auto newNode =
      std::make_unique<DbAcquireNode>(op, this, this, analysis, childId);
  DbAcquireNode *ptr = newNode.get();
  acquireNodes.push_back(std::move(newNode));
  acquireMap[op] = ptr;
  return ptr;
}
