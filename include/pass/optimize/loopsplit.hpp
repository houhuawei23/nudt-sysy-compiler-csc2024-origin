#pragma once
#include <set>
#include <cassert>
#include <map>
#include <vector>
#include "ir/ir.hpp"
#include "pass/pass.hpp"
namespace pass {

class loopsplit : public FunctionPass {
  private:
    DomTree* domctx;
    sideEffectInfo* sectx;
    LoopInfo* lpctx;
    IndVarInfo* ivctx;
    TopAnalysisInfoManager* tpctx;
    ir::BranchInst* brinst = nullptr;
    ir::ICmpInst* icmpinst = nullptr;
    ir::Value* endval = nullptr;
    ir::BasicBlock* condbb = nullptr;

    ir::PhiInst* ivphi = nullptr;
    ir::ICmpInst* ivicmp = nullptr;
    ir::BinaryInst* iviter = nullptr;

  public:
    // bool getinfo(ir::Loop* L);
    void splitloop(ir::Loop* L);
    bool dosplit(ir::Function* func, TopAnalysisInfoManager* tp);
    bool couldsplit(ir::Loop* loop);
    std::string name() const override { return "loopsplit"; }
    void run(ir::Function* func, TopAnalysisInfoManager* tp) override;
};
}  // namespace pass