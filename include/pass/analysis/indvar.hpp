#pragma once
#include "pass/pass.hpp"

namespace pass {
class indVarAnalysis : public FunctionPass {
public:
  std::string name() const override { return "indVarAnalysis"; }
  void run(ir::Function* func, TopAnalysisInfoManager* tp) override;

private:
  LoopInfo* lpctx;
  IndVarInfo* ivctx;
  DomTree* domctx;
  sideEffectInfo* sectx;
  void addIndVar(ir::Loop* lp,
                 ir::Value* mbegin,
                 ir::Value* mstep,
                 ir::Value* mend,
                 ir::BinaryInst* iterinst,
                 ir::Instruction* cmpinst,
                 ir::PhiInst* phiinst);
  ir::ConstantInteger* getConstantBeginVarFromPhi(ir::PhiInst* phiinst,
                                                  ir::PhiInst* oldPhiinst,
                                                  ir::Loop* lp);
  bool isSimplyNotInLoop(ir::Loop* lp,ir::Value* val);
  bool isSimplyLoopInvariant(ir::Loop* lp,ir::Value* val);
};

class indVarInfoCheck : public FunctionPass {
public:
  void run(ir::Function* func, TopAnalysisInfoManager* tp) override;
  std::string name() const override { return "indVarCheckInfo"; }

private:
  LoopInfo* lpctx;
  IndVarInfo* ivctx;
};
}  // namespace pass