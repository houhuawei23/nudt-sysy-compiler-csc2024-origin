#pragma once
#include <set>
#include <cassert>
#include <map>
#include <vector>
#include <algorithm>
#include <queue>
#include "ir/ir.hpp"
#include "pass/pass.hpp"

namespace pass {
class tailCallOpt : public FunctionPass {
public:
  std::string name() const override { return "tailCallOpt"; }
  void run(ir::Function* func, TopAnalysisInfoManager* tp) override;

private:
  bool is_tail_rec(ir::Instruction* inst, ir::Function* func);
  bool is_tail_call(ir::Instruction* inst, ir::Function* func);
  void recursiveDeleteInst(ir::Instruction* inst);
  void recursiveDeleteBB(ir::BasicBlock* bb);
};
}  // namespace pass