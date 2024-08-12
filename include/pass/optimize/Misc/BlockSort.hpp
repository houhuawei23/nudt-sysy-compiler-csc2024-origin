#pragma once
#include "ir/ir.hpp"
#include "pass/pass.hpp"
#include "pass/optimize/Utils/BlockUtils.hpp"
using namespace ir;

namespace pass {
class BlockSort : public FunctionPass {
  bool runImpl(ir::Function* func, TopAnalysisInfoManager* tp) { return blockSortDFS(*func, tp); }

public:
  void run(ir::Function* func, TopAnalysisInfoManager* tp) override { runImpl(func, tp); }
  std::string name() const override { return "BlockSort"; }
};
}  // namespace pass