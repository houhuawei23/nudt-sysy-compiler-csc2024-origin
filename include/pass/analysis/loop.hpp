#pragma once
#include "pass/pass.hpp"

namespace pass {
class loopAnalysis : public FunctionPass {
public:
  void run(ir::Function* func, TopAnalysisInfoManager* tp) override;
  std::string name() const override { return "Loop Analysis"; }

private:
  void addLoopBlocks(ir::Function* func,
                     ir::BasicBlock* header,
                     ir::BasicBlock* tail);
  void loopGetExits(ir::Loop* plp);
  LoopInfo* lpctx;
  DomTree* domctx;
};

class loopInfoCheck : public FunctionPass {
public:
  void run(ir::Function* func, TopAnalysisInfoManager* tp) override;
  std::string name() const override { return "Loop Info Check"; }

private:
  LoopInfo* lpctx;
};
}  // namespace pass
