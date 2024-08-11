#pragma once
#include "ir/ir.hpp"
#include "pass/analysisinfo.hpp"
#include <functional>

using namespace ir;
using BlockReducer = std::function<ir::Value*(ir::Instruction* inst)>;

namespace pass {
bool reduceBlock(IRBuilder& builder, BasicBlock& block, const BlockReducer& reducer);

BasicBlock* splitBlock(BasicBlockList& blocks,
                       BasicBlockList::iterator blockIt,
                       InstructionList::iterator instIt);

bool fixAllocaInEntry(Function& func);

bool blockSort(Function& func, TopAnalysisInfoManager* tAIM);

}  // namespace pass