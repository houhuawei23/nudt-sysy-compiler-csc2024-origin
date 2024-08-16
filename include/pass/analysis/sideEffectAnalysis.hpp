#pragma once
#include "pass/pass.hpp"
using namespace pass;

class sideEffectAnalysis : public ModulePass {
public:
  std::string name() const override { return "sideEffectAnalysis"; }
  void run(ir::Module* md, TopAnalysisInfoManager* tp) override;
private:
  void infoCheck(ir::Module* md);
  ir::Value* getBaseAddr(ir::Value* subAddr);
  bool propogateSideEffect(ir::Module* md);


private:
  callGraph* cgctx;
  sideEffectInfo* sectx;
};
