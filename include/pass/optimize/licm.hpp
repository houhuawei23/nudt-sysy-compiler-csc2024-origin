#pragma once
#include <set>
#include <cassert>
#include <map>
#include <vector>
#include "ir/ir.hpp"
#include "pass/pass.hpp"
namespace pass {

class LICM : public FunctionPass {
  private:
    sideEffectInfo* sectx;
    loopInfo* loopctx;
    domTree* domctx;
    pdomTree* pdomcctx;
  public:
    std::string name() const override { return "LICM"; }
    bool checkstore(ir::LoadInst* loadinst, ir::Loop* loop);
    bool checkload(ir::StoreInst* storeinst, ir::Loop* loop);
    bool alias(ir::Instruction* inst0, ir::Instruction* inst1);
    ir::Value* getbase(ir::Instruction* inst);
    bool safestore(ir::StoreInst* safestore, ir::Loop* loop);
    bool isinvariantop(ir::Instruction* inst, ir::Loop* loop);
    std::vector<ir::Instruction*> getinvariant(ir::BasicBlock* bb, ir::Loop* loop);
    void run(ir::Function* func, TopAnalysisInfoManager* tp) override;
};
}  // namespace pass