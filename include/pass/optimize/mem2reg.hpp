#pragma once
#include <set>
#include <cassert>
#include <map>
#include <vector>
#include "ir/ir.hpp"
#include "pass/pass.hpp"

namespace pass {
class Mem2Reg : public FunctionPass {
  private:
    domTree* domctx;
    std::vector<ir::AllocaInst*> Allocas;
    std::map<ir::AllocaInst*, std::set<ir::BasicBlock*>> DefsBlock;
    std::map<ir::AllocaInst*, std::set<ir::BasicBlock*>> UsesBlock;

    std::map<ir::BasicBlock*, std::map<ir::PhiInst*, ir::AllocaInst*>> PhiMap;
    std::map<ir::AllocaInst*, ir::Argument*> ValueMap;
    std::vector<ir::PhiInst*> allphi;

  public:
    void run(ir::Function* func, TopAnalysisInfoManager* tp) override;

    void promotememToreg(ir::Function* F);
    void RemoveFromAllocasList(unsigned& AllocaIdx);
    void allocaAnalysis(ir::AllocaInst* alloca);
    bool promotemem2reg(ir::Function* F);
    bool is_promoted(ir::AllocaInst* alloca);
    void insertphi();
    void rename(ir::Function* F);
    void simplifyphi(ir::PhiInst* phi);
    bool coulddelete(ir::AllocaInst* alloca);
};
}  // namespace pass
