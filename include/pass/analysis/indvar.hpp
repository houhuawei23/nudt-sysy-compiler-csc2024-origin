#pragma once
#include "pass/pass.hpp"

namespace pass {
class indVarAnalysis : public FunctionPass {
public:
  std::string name() const override { return "indVarAnalysis"; }
  void run(ir::Function* func, TopAnalysisInfoManager* tp) override;

private:
  loopInfo* lpctx;
  indVarInfo* ivctx;
  void addIndVar(ir::Loop* lp,
                 ir::Constant* mbegin,
                 ir::Constant* mstep,
                 ir::Value* mend,
                 ir::BinaryInst* iterinst,
                 ir::Instruction* cmpinst,
                 ir::PhiInst* phiinst);
  ir::Constant* getConstantBeginVarFromPhi(ir::PhiInst* phiinst,ir::PhiInst* oldPhiinst,ir::Loop* lp);
};

class indVarInfoCheck : public FunctionPass {
public:
  void run(ir::Function* func, TopAnalysisInfoManager* tp) override;
  std::string name() const override { return "indVarCheckInfo"; }

private:
  loopInfo* lpctx;
  indVarInfo* ivctx;
};
}  // namespace pass