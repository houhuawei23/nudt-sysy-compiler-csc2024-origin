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
    TopAnalysisInfoManager* tpm;
    std::set<ir::Value*> loopStorePtrs;

  public:
    std::string name() const override { return "LICM"; }
    bool checkstore(ir::LoadInst* loadinst, ir::Loop* loop);
    bool checkload(ir::StoreInst* storeinst, ir::Loop* loop);
    bool alias(ir::Instruction* inst0, ir::Instruction* inst1);
    ir::Value* getIntToPtrBaseAddr(ir::UnaryInst* inst);
    ir::Value* getbase(ir::Value* val);
    bool iswrite(ir::Value* ptr, ir::CallInst* callinst);
    bool isread(ir::Value* ptr, ir::CallInst* callinst);
    void collectStorePtrs(ir::CallInst* call, ir::Loop* loop);
    bool isinvariantcall(ir::CallInst* callinst, ir::Loop* loop);
    bool safestore(ir::StoreInst* safestore, ir::Loop* loop);
    bool isinvariantop(ir::Instruction* inst, ir::Loop* loop);
    std::vector<ir::Instruction*> getinvariant(ir::BasicBlock* bb, ir::Loop* loop);
    void run(ir::Function* func, TopAnalysisInfoManager* tp) override;
};
}  // namespace pass