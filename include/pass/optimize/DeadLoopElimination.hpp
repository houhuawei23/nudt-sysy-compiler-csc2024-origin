#pragma once
#include <set>
#include <cassert>
#include <map>
#include <vector>
#include "ir/ir.hpp"
#include "pass/pass.hpp"
namespace pass {

class DeadLoopElimination : public FunctionPass {
  private:
    domTree* domctx;
    sideEffectInfo* sectx;
    loopInfo* lpctx;
    indVarInfo* ivctx;

  public:
    bool isDeadLoop(ir::IndVar* iv, ir::Loop* loop);
    void deleteDeadLoop(ir::Loop* loop);
    std::string name() const override { return "DeadLoop"; }
    void run(ir::Function* func, TopAnalysisInfoManager* tp) override;
};
}  // namespace pass