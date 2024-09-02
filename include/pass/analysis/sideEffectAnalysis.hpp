#pragma once
#include "pass/pass.hpp"
using namespace pass;
namespace pass{
class sideEffectAnalysis : public ModulePass {
public:
  std::string name() const override { return "sideEffectAnalysis"; }
  void run(ir::Module* md, TopAnalysisInfoManager* tp) override;

private:
  void infoCheck(ir::Module* md);
  bool propogateSideEffect(ir::Module* md);
  TopAnalysisInfoManager* topmana;
  ir::Value* getIntToPtrBaseAddr(ir::UnaryInst* inst);
  ir::Value* getBaseAddr(ir::Value* subAddr);

private:
  CallGraph* cgctx;
  sideEffectInfo* sectx;
};
}

