#pragma once
#include <vector>
#include "ir/ir.hpp"
#include "pass/analysisinfo.hpp"

namespace pass {

//! Pass Template
template <typename PassUnit>
class Pass {
  public:
  // pure virtual function, define the api
  virtual void run(PassUnit* pass_unit, TopAnalysisInfoManager* tp) = 0;
};

// Instantiate Pass Class for Module, Function and BB
using ModulePass = Pass<ir::Module>;
using FunctionPass = Pass<ir::Function>;
using BasicBlockPass = Pass<ir::BasicBlock>;

class PassManager {
  ir::Module* irModule;
  pass::TopAnalysisInfoManager* tAIM;

  public:
  PassManager(ir::Module* pm, TopAnalysisInfoManager* tp) {
    irModule = pm;
    tAIM = tp;
  }
  void run(ModulePass* mp) { mp->run(irModule, tAIM); }
  void run(FunctionPass* fp) {
    for (auto func : irModule->funcs()) {
      if (func->isOnlyDeclare()) continue;
      fp->run(func, tAIM);
    }
  }
  void run(BasicBlockPass* bp) {
    for (auto func : irModule->funcs()) {
      for (auto bb : func->blocks()) {
        bp->run(bb, tAIM);
      }
    }
  }
  void runPasses(std::vector<std::string> passes);

};

class TopAnalysisInfoManager {
  private:
  ir::Module* mModule;
  // ir::Module info
  callGraph* mCallGraph;
  // ir::Function info
  std::unordered_map<ir::Function*, domTree*> mDomTree;
  std::unordered_map<ir::Function*, pdomTree*> mPDomTree;
  std::unordered_map<ir::Function*, loopInfo*> mLoopInfo;
  std::unordered_map<ir::Function*, indVarInfo*> mIndVarInfo;
  // bb info
  public:
  TopAnalysisInfoManager(ir::Module* pm) : mModule(pm), mCallGraph(nullptr) {}
  domTree* getDomTree(ir::Function* func) {
    if (func->isOnlyDeclare()) return nullptr;
    return mDomTree[func];
  }
  pdomTree* getPDomTree(ir::Function* func) {
    if (func->isOnlyDeclare()) return nullptr;
    return mPDomTree[func];
  }
  loopInfo* getLoopInfo(ir::Function* func) {
    if (func->isOnlyDeclare()) return nullptr;
    return mLoopInfo[func];
  }
  indVarInfo* getIndVarInfo(ir::Function* func) { 
    if (func->isOnlyDeclare()) return nullptr;
    return mIndVarInfo[func]; 
  }

  callGraph* getCallGraph() { return mCallGraph; }
  void initialize();
  void CFGChange(ir::Function* func) {
    if (func->isOnlyDeclare()) return;
    mDomTree[func]->setOff();
    mPDomTree[func]->setOff();
    mLoopInfo[func]->setOff();
  }
  void CallChange() { mCallGraph->setOff(); }
  void IndVarChange(ir::Function* func) { 
    if (func->isOnlyDeclare()) return;
    mIndVarInfo[func]->setOff(); 
  }
};

}  // namespace pass