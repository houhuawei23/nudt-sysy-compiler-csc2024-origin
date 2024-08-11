#pragma once
#include <vector>
#include "ir/ir.hpp"
#include "pass/analysisinfo.hpp"

#include <chrono>
namespace pass {

//! Pass Template
template <typename PassUnit>
class Pass {
public:
  // pure virtual function, define the api
  virtual void run(PassUnit* pass_unit, TopAnalysisInfoManager* tp) = 0;
  virtual std::string name() const = 0;
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
  void run(ModulePass* mp);
  void run(FunctionPass* fp);
  void run(BasicBlockPass* bp);
  void runPasses(std::vector<std::string> passes);
};

class TopAnalysisInfoManager {
private:
  ir::Module* mModule;
  // ir::Module info
  callGraph* mCallGraph;
  sideEffectInfo* mSideEffectInfo;
  // ir::Function info
  std::unordered_map<ir::Function*, domTree*> mDomTree;
  std::unordered_map<ir::Function*, pdomTree*> mPDomTree;
  std::unordered_map<ir::Function*, loopInfo*> mLoopInfo;
  std::unordered_map<ir::Function*, indVarInfo*> mIndVarInfo;
  std::unordered_map<ir::Function*, dependenceInfoForLoops*> mDepInfo;
  // bb info
public:
  TopAnalysisInfoManager(ir::Module* pm) : mModule(pm), mCallGraph(nullptr) {}
  domTree* getDomTree(ir::Function* func) {
    if (func->isOnlyDeclare()) return nullptr;
    auto domctx=mDomTree[func];
    domctx->refresh();
    return domctx;
  }
  pdomTree* getPDomTree(ir::Function* func) {
    if (func->isOnlyDeclare()) return nullptr;
    auto domctx=mPDomTree[func];
    domctx->refresh();    
    return domctx;
  }
  loopInfo* getLoopInfo(ir::Function* func) {
    if (func->isOnlyDeclare()) return nullptr;
    auto lpctx=mLoopInfo[func];
    lpctx->refresh();
    return lpctx;
  }
  indVarInfo* getIndVarInfo(ir::Function* func) {
    if (func->isOnlyDeclare()) return nullptr;
    auto idvctx=mIndVarInfo[func];
    idvctx->setOff();
    idvctx->refresh();
    return idvctx;
  }
  dependenceInfoForLoops* getDepInfo(ir::Function* func){
    if (func->isOnlyDeclare()) return nullptr;
    auto dpctx=mDepInfo[func];
    dpctx->setOff();
    dpctx->refresh();
    return dpctx;
  }


  callGraph* getCallGraph() { 
    mCallGraph->refresh();
    return mCallGraph; 
  }
  sideEffectInfo* getSideEffectInfo(){
    mSideEffectInfo->setOff(); 
    mSideEffectInfo->refresh();
    return mSideEffectInfo;
  }
  
  
  domTree* getDomTreeWithoutRefresh(ir::Function* func) {
    if (func->isOnlyDeclare()) return nullptr;
    auto domctx=mDomTree[func];
    return domctx;
  }
  pdomTree* getPDomTreeWithoutRefresh(ir::Function* func) {
    if (func->isOnlyDeclare()) return nullptr;
    auto domctx=mPDomTree[func];
    return domctx;
  }
  loopInfo* getLoopInfoWithoutRefresh(ir::Function* func) {
    if (func->isOnlyDeclare()) return nullptr;
    auto lpctx=mLoopInfo[func];
    return lpctx;
  }
  indVarInfo* getIndVarInfoWithoutRefresh(ir::Function* func) {
    if (func->isOnlyDeclare()) return nullptr;
    auto idvctx=mIndVarInfo[func];
    return idvctx;
  }
  dependenceInfoForLoops* getDepInfoWithoutRefresh(ir::Function* func) {
    if (func->isOnlyDeclare()) return nullptr;
    auto dpctx=mDepInfo[func];
    return dpctx;
  }


  callGraph* getCallGraphWithoutRefresh() { 
    return mCallGraph; 
  }
  sideEffectInfo* getSideEffectInfoWithoutRefresh(){
    return mSideEffectInfo;
  }

  void initialize();
  void CFGChange(ir::Function* func) {
    if (func->isOnlyDeclare()) return;
    if(mDomTree.find(func) == mDomTree.cend()) {
      std::cerr << "DomTree not found for function " << func->name() << std::endl;
      return;
    }
    mDomTree[func]->setOff();
    mPDomTree[func]->setOff();
    mLoopInfo[func]->setOff();
    mIndVarInfo[func]->setOff();
  }
  void CallChange() { mCallGraph->setOff(); }
  void IndVarChange(ir::Function* func) {
    if (func->isOnlyDeclare()) return;
    mIndVarInfo[func]->setOff();
  }
};

}  // namespace pass