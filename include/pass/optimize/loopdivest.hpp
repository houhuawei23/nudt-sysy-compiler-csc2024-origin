#pragma once
#include <set>
#include <cassert>
#include <map>
#include <vector>
#include "ir/ir.hpp"
#include "pass/pass.hpp"
#include "pass/optimize/loopunroll.hpp"
namespace pass {

class loopdivest : public FunctionPass {
  private:
    DomTree* domctx;
    sideEffectInfo* sectx;
    LoopInfo* lpctx;
    IndVarInfo* ivctx;

  public:
    std::string name() const override { return "loopdivest"; }
    bool shoulddivest(ir::Loop* loop);
    void runonloop(ir::Loop* loop, ir::Function* func);
    void run(ir::Function* func, TopAnalysisInfoManager* tp) override;
};
}  // namespace pass