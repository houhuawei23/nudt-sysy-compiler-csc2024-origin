#pragma once
#include <set>
#include <cassert>
#include <map>
#include <vector>
#include "ir/ir.hpp"
#include "pass/pass.hpp"

namespace pass {
class GCM : public FunctionPass {
private:
  std::set<ir::Instruction*> insts_visited;

public:
  void run(ir::Function* func, TopAnalysisInfoManager* tp) override;
  DomTree* domctx;
  LoopInfo* lpctx;
  sideEffectInfo* sectx;
  void scheduleEarly(ir::Instruction* instruction, ir::BasicBlock* entry);
  // void scheduleLate(ir::Instruction *instruction, ir::BasicBlock* exit);
  ir::BasicBlock* LCA(ir::BasicBlock* lhs, ir::BasicBlock* rhs);
  bool ispinned(ir::Instruction* instruction);

  std::string name() const override { return "GCM"; }
};
}  // namespace pass