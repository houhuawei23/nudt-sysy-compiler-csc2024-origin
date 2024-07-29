#pragma once
#include "pass/pass.hpp"
using namespace pass;

class sideEffectAnalysis : public ModulePass {
public:
  std::string name() const override { return "sideEffectAnalysis"; }
  void run(ir::Module* md, TopAnalysisInfoManager* tp) override;
  void infoCheck(ir::Module* md);

private:
  callGraph* cgctx;
  sideEffectInfo* sectx;
};
