#pragma once
#include "ir/ir.hpp"
#include <functional>

using namespace ir;
using BlockReducer = std::function<ir::Value*(ir::Instruction* inst)>;

namespace pass {
bool reduceBlock(IRBuilder& builder, BasicBlock& block, const BlockReducer& reducer);

BasicBlock* splitBlock(BasicBlockList& blocks,
                       BasicBlockList::iterator blockIt,
                       InstructionList::iterator instIt);
}  // namespace pass