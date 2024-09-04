#pragma once
// add analysis passes
#include "ir/ir.hpp"
#include <unordered_map>
#include <vector>
#include <queue>

using namespace ir;
namespace pass {
template <typename PassUnit>
class AnalysisInfo;

class DomTree;
class PDomTree;
class LoopInfo;
class CallGraph;
class IndVarInfo;
class TopAnalysisInfoManager;
class DependenceAnalysis;
class LoopDependenceInfo;

template <typename PassUnit>
class AnalysisInfo {
protected:
  PassUnit* passUnit;
  TopAnalysisInfoManager* topManager;
  bool isValid;

public:
  AnalysisInfo(PassUnit* mp, TopAnalysisInfoManager* mtp, bool v = false)
    : isValid(v), passUnit(mp), topManager(mtp) {}
  void setOn() { isValid = true; }
  void setOff() { isValid = false; }
  virtual void refresh() = 0;
};
using ModuleACtx = AnalysisInfo<Module>;
using FunctionACtx = AnalysisInfo<Function>;

// add new analysis info of ir here!
// dom Tree
/*
Dominate Tree
idom: immediate dominator
sdom: strict dominator
*/
class DomTree : public FunctionACtx {
protected:
  std::unordered_map<BasicBlock*, BasicBlock*> _idom;
  std::unordered_map<BasicBlock*, BasicBlock*> _sdom;
  std::unordered_map<BasicBlock*, int> _domlevel;
  std::unordered_map<BasicBlock*, std::vector<BasicBlock*>> _domson;
  std::unordered_map<BasicBlock*, std::vector<BasicBlock*>> _domfrontier;
  std::vector<BasicBlock*> _BFSDomTreeVector;
  std::vector<BasicBlock*> _DFSDomTreeVector;

public:
  DomTree(Function* func, TopAnalysisInfoManager* tp) : FunctionACtx(func, tp) {}
  BasicBlock* idom(BasicBlock* bb) { return _idom[bb]; }
  void set_idom(BasicBlock* bb, BasicBlock* idbb) { _idom[bb] = idbb; }
  BasicBlock* sdom(BasicBlock* bb) { return _sdom[bb]; }
  void set_sdom(BasicBlock* bb, BasicBlock* sdbb) { _sdom[bb] = sdbb; }
  int domlevel(BasicBlock* bb) { return _domlevel[bb]; }
  void set_domlevel(BasicBlock* bb, int lv) { _domlevel[bb] = lv; }

  auto& domson(BasicBlock* bb) { return _domson[bb]; }

  auto& domfrontier(BasicBlock* bb) { return _domfrontier[bb]; }

  auto& BFSDomTreeVector() { return _BFSDomTreeVector; }

  auto& DFSDomTreeVector() { return _DFSDomTreeVector; }

  void clearAll() {
    _idom.clear();
    _sdom.clear();
    _domson.clear();
    _domfrontier.clear();
    _domlevel.clear();
    _BFSDomTreeVector.clear();
    _DFSDomTreeVector.clear();
  }
  void initialize() {
    clearAll();
    for (auto bb : passUnit->blocks()) {
      _domson[bb] = std::vector<BasicBlock*>();
      _domfrontier[bb] = std::vector<BasicBlock*>();
    }
  }
  void refresh() override;
  bool dominate(BasicBlock* bb1, BasicBlock* bb2) {
    if (bb1 == bb2) return true;
    auto bbIdom = _idom[bb2];
    while (bbIdom != nullptr) {
      if (bbIdom == bb1) return true;
      bbIdom = _idom[bbIdom];
    }
    return false;
  }
  void BFSDomTreeInfoRefresh() {
    std::queue<BasicBlock*> bbqueue;
    std::unordered_map<BasicBlock*, bool> vis;
    for (auto bb : passUnit->blocks())
      vis[bb] = false;

    _BFSDomTreeVector.clear();
    bbqueue.push(passUnit->entry());

    while (!bbqueue.empty()) {
      auto bb = bbqueue.front();
      bbqueue.pop();
      if (!vis[bb]) {
        _BFSDomTreeVector.push_back(bb);
        vis[bb] = true;
        for (auto bbDomSon : _domson[bb])
          bbqueue.push(bbDomSon);
      }
    }
  }
  void DFSDomTreeInfoRefresh() {
    std::stack<BasicBlock*> bbstack;
    std::unordered_map<BasicBlock*, bool> vis;
    for (auto bb : passUnit->blocks())
      vis[bb] = false;

    _DFSDomTreeVector.clear();
    bbstack.push(passUnit->entry());

    while (!bbstack.empty()) {
      auto bb = bbstack.top();
      bbstack.pop();
      if (!vis[bb]) {
        _DFSDomTreeVector.push_back(bb);
        vis[bb] = true;
        for (auto bbDomSon : _domson[bb])
          bbstack.push(bbDomSon);
      }
    }
  }
};

class PDomTree : public FunctionACtx {  // also used as pdom
protected:
  std::unordered_map<BasicBlock*, BasicBlock*> _ipdom;
  std::unordered_map<BasicBlock*, BasicBlock*> _spdom;
  std::unordered_map<BasicBlock*, int> _pdomlevel;
  std::unordered_map<BasicBlock*, std::vector<BasicBlock*>> _pdomson;
  std::unordered_map<BasicBlock*, std::vector<BasicBlock*>> _pdomfrontier;

public:
  PDomTree(Function* func, TopAnalysisInfoManager* tp) : FunctionACtx(func, tp) {}
  BasicBlock* ipdom(BasicBlock* bb) {
    assert(bb && "bb is null");
    return _ipdom[bb];
  }
  void set_ipdom(BasicBlock* bb, BasicBlock* idbb) { _ipdom[bb] = idbb; }
  BasicBlock* spdom(BasicBlock* bb) {
    assert(bb && "bb is null");
    return _spdom[bb];
  }
  void set_spdom(BasicBlock* bb, BasicBlock* sdbb) { _spdom[bb] = sdbb; }
  int pdomlevel(BasicBlock* bb) {
    assert(bb && "bb is null");
    return _pdomlevel[bb];
  }
  void set_pdomlevel(BasicBlock* bb, int lv) { _pdomlevel[bb] = lv; }
  std::vector<BasicBlock*>& pdomson(BasicBlock* bb) { return _pdomson[bb]; }
  std::vector<BasicBlock*>& pdomfrontier(BasicBlock* bb) { return _pdomfrontier[bb]; }
  void clearAll() {
    _ipdom.clear();
    _spdom.clear();
    _pdomson.clear();
    _pdomfrontier.clear();
    _pdomlevel.clear();
  }
  void initialize() {
    clearAll();
    for (auto bb : passUnit->blocks()) {
      _pdomson[bb] = std::vector<BasicBlock*>();
      _pdomfrontier[bb] = std::vector<BasicBlock*>();
    }
  }

  bool pdominate(BasicBlock* bb1, BasicBlock* bb2) {
    if (bb1 == bb2) return true;
    auto bbIdom = _ipdom[bb2];
    while (bbIdom != nullptr) {
      if (bbIdom == bb1) return true;
      bbIdom = _ipdom[bbIdom];
    }
    return false;
  }

  void refresh() override;
};

class LoopInfo : public FunctionACtx {
protected:
  std::vector<Loop*> _loops;
  std::unordered_map<BasicBlock*, Loop*> _head2loop;
  std::unordered_map<BasicBlock*, size_t> _looplevel;

public:
  LoopInfo(Function* fp, TopAnalysisInfoManager* tp) : FunctionACtx(fp, tp) {}
  std::vector<Loop*>& loops() { return _loops; }
  Loop* head2loop(BasicBlock* bb) {
    if (_head2loop.count(bb) == 0) return nullptr;
    return _head2loop[bb];
  }
  void set_head2loop(BasicBlock* bb, Loop* lp) { _head2loop[bb] = lp; }
  int looplevel(BasicBlock* bb) { return _looplevel[bb]; }
  void set_looplevel(BasicBlock* bb, int lv) { _looplevel[bb] = lv; }
  void clearAll() {
    _loops.clear();
    _head2loop.clear();
    _looplevel.clear();
  }
  bool isHeader(BasicBlock* bb) { return _head2loop.count(bb); }
  Loop* getinnermostLoop(BasicBlock* bb) {  // 返回最内层的循环
    Loop* innermost = nullptr;
    for (auto L : _loops) {
      if (L->contains(bb)) {
        if (innermost == nullptr)
          innermost = L;
        else {
          if (_looplevel[L->header()] < _looplevel[innermost->header()]) innermost = L;
        }
      }
    }
    return innermost;
  }
  void refresh() override;
  void print(std::ostream& os) const;
  std::vector<Loop*> sortedLoops(bool reverse = false);  // looplevel small to big
};

class CallGraph : public ModuleACtx {
protected:
  std::unordered_map<Function*, std::set<Function*>> _callees;
  std::unordered_map<Function*, std::set<Function*>> _callers;
  std::unordered_map<Function*, bool> _is_called;
  std::unordered_map<Function*, bool> _is_inline;
  std::unordered_map<Function*, bool> _is_lib;
  std::unordered_map<Function*, std::set<CallInst*>>
    _callerCallInsts;  // func's caller insts are func's callers'
  std::unordered_map<Function*, std::set<CallInst*>>
    _calleeCallInsts;  // func's callee insts are func's

public:
  CallGraph(Module* md, TopAnalysisInfoManager* tp) : ModuleACtx(md, tp) {}
  std::set<Function*>& callees(Function* func) { return _callees[func]; }
  std::set<Function*>& callers(Function* func) { return _callers[func]; }
  std::set<CallInst*>& callerCallInsts(Function* func) { return _callerCallInsts[func]; }
  std::set<CallInst*>& calleeCallInsts(Function* func) { return _calleeCallInsts[func]; }
  bool isCalled(Function* func) { return _is_called[func]; }
  bool isInline(Function* func) { return _is_inline[func]; }
  bool isLib(Function* func) { return _is_lib[func]; }
  void set_isCalled(Function* func, bool b) { _is_called[func] = b; }
  void set_isInline(Function* func, bool b) { _is_inline[func] = b; }
  void set_isLib(Function* func, bool b) { _is_lib[func] = b; }
  void clearAll() {
    _callees.clear();
    _callers.clear();
    _is_called.clear();
    _is_inline.clear();
    _is_lib.clear();
    _callerCallInsts.clear();
    _calleeCallInsts.clear();
  }
  void initialize() {
    for (auto func : passUnit->funcs()) {
      _callees[func] = std::set<Function*>();
      _callers[func] = std::set<Function*>();
    }
  }
  bool isNoCallee(Function* func) {
    if (_callees[func].size() == 0) return true;
    for (auto f : _callees[func]) {
      if (not isLib(f)) return false;
    }
    return true;
  }
  void refresh() override;
};

class IndVarInfo : public FunctionACtx {
private:
  std::unordered_map<Loop*, IndVar*> _loopToIndvar;

public:
  IndVarInfo(Function* fp, TopAnalysisInfoManager* tp) : FunctionACtx(fp, tp) {}
  IndVar* getIndvar(Loop* loop) {
    if (_loopToIndvar.count(loop) == 0) return nullptr;
    return _loopToIndvar.at(loop);
  }
  void clearAll() { _loopToIndvar.clear(); }
  void refresh() override;
  void addIndVar(Loop* lp, IndVar* idv) { _loopToIndvar[lp] = idv; }
};

class SideEffectInfo : public ModuleACtx {
private:
  // 当前函数读取的全局变量
  std::unordered_map<Function*, std::set<GlobalVariable*>> _FuncReadGlobals;
  // 当前函数写入的全局变量
  std::unordered_map<Function*, std::set<GlobalVariable*>> _FuncWriteGlobals;
  // 对于当前argument函数是否读取（仅限pointer）
  std::unordered_map<Argument*, bool> _isArgumentRead;
  // 对于当前argument哈数是否写入（仅限pointer）
  std::unordered_map<Argument*, bool> _isArgumentWrite;
  std::unordered_map<Function*, bool> _isLib;  // 当前函数是否为lib函数
  // 当前函数的参数中有哪些是指针参数
  std::unordered_map<Function*, std::set<Argument*>> _funcPointerArgs;
  // 当前函数有无调用库函数或者简介调用库函数
  std::unordered_map<Function*, bool> _isCallLibFunc;
  // 出现了无法分析基址的情况，含有潜在的副作用
  std::unordered_map<Function*, bool> _hasPotentialSideEffect;
  // 当前函数直接读取的gv
  std::unordered_map<Function*, std::set<GlobalVariable*>> _FuncReadDirectGvs;
  // 当前函数直接写入的gv
  std::unordered_map<Function*, std::set<GlobalVariable*>> _FuncWriteDirectGvs;

public:
  SideEffectInfo(Module* ctx, TopAnalysisInfoManager* tp) : ModuleACtx(ctx, tp) {}
  void clearAll() {
    _FuncReadGlobals.clear();
    _FuncWriteGlobals.clear();
    _isArgumentRead.clear();
    _isArgumentWrite.clear();
    _isLib.clear();
    _funcPointerArgs.clear();
    _isCallLibFunc.clear();
    _hasPotentialSideEffect.clear();
    _FuncReadDirectGvs.clear();
    _FuncWriteDirectGvs.clear();
  }
  void refresh() override;
  // get
  bool getArgRead(Argument* arg) { return _isArgumentRead[arg]; }
  bool getArgWrite(Argument* arg) { return _isArgumentWrite[arg]; }
  bool getIsLIb(Function* func) { return _isLib[func]; }
  bool getIsCallLib(Function* func) { return _isCallLibFunc[func]; }
  bool getPotentialSideEffect(Function* func) { return _hasPotentialSideEffect[func]; }
  // set
  void setArgRead(Argument* arg, bool b) { _isArgumentRead[arg] = b; }
  void setArgWrite(Argument* arg, bool b) { _isArgumentWrite[arg] = b; }
  void setFuncIsLIb(Function* func, bool b) { _isLib[func] = b; }
  void setFuncIsCallLib(Function* func, bool b) { _isCallLibFunc[func] = b; }
  void setPotentialSideEffect(Function* func, bool b) { _hasPotentialSideEffect[func] = b; }
  // reference
  std::set<GlobalVariable*>& funcReadGlobals(Function* func) { return _FuncReadGlobals[func]; }
  std::set<GlobalVariable*>& funcWriteGlobals(Function* func) { return _FuncWriteGlobals[func]; }
  std::set<Argument*>& funcArgSet(Function* func) { return _funcPointerArgs[func]; }
  std::set<GlobalVariable*>& funcDirectReadGvs(Function* func) { return _FuncReadDirectGvs[func]; }
  std::set<GlobalVariable*>& funcDirectWriteGvs(Function* func) {
    return _FuncWriteDirectGvs[func];
  }
  // old API
  bool hasSideEffect(Function* func) {
    if (_isLib[func]) return true;
    if (_isCallLibFunc[func]) return true;
    if (not _FuncWriteGlobals[func].empty()) return true;
    if (_hasPotentialSideEffect[func]) return true;

    for (auto arg : _funcPointerArgs[func]) {
      if (getArgWrite(arg)) return true;
    }
    return false;
  }
  bool isPureFunc(Function* func) {
    for (auto arg : funcArgSet(func)) {
      if (getArgRead(arg)) return false;
    }
    return (not hasSideEffect(func) and _FuncReadGlobals[func].empty()) and not _isLib[func];
  }
  bool isInputOnlyFunc(Function* func) {
    if (hasSideEffect(func)) return false;
    if (not _FuncReadGlobals[func].empty()) return false;
    return true;
  }
  void functionInit(Function* func) {
    _FuncReadGlobals[func] = std::set<GlobalVariable*>();
    _FuncWriteGlobals[func] = std::set<GlobalVariable*>();
    _FuncWriteDirectGvs[func] = std::set<GlobalVariable*>();
    _FuncReadDirectGvs[func] = std::set<GlobalVariable*>();
    for (auto arg : func->args()) {
      if (not arg->isPointer()) continue;
      _funcPointerArgs[func].insert(arg);
      setArgRead(arg, false);
      setArgWrite(arg, false);
    }
    _isCallLibFunc[func] = false;
  }
};

class DependenceInfo : public FunctionACtx {
private:
  std::unordered_map<Loop*, LoopDependenceInfo*> mFunc2LoopDepInfo;

public:
  DependenceInfo(Function* func, TopAnalysisInfoManager* tp) : FunctionACtx(func, tp) {}
  LoopDependenceInfo* getLoopDependenceInfo(Loop* lp) {
    if (mFunc2LoopDepInfo.count(lp))
      return mFunc2LoopDepInfo[lp];
    else
      return nullptr;
  }
  void clearAll() { mFunc2LoopDepInfo.clear(); }
  void refresh() override;
  void setDepInfoLp(Loop* lp, LoopDependenceInfo* input) { mFunc2LoopDepInfo[lp] = input; }
};

class ParallelInfo : public FunctionACtx {
  // 你想并行的这里都有!
private:
  std::unordered_map<BasicBlock*, bool> mLpIsParallel;
  std::unordered_map<BasicBlock*, std::set<PhiInst*>> mLpPhis;
  std::unordered_map<PhiInst*, bool> mIsPhiAdd;
  std::unordered_map<PhiInst*, bool> mIsPhiSub;
  std::unordered_map<PhiInst*, bool> mIsPhiMul;
  std::unordered_map<PhiInst*, Value*> mModuloVal;

public:
  ParallelInfo(Function* func, TopAnalysisInfoManager* tp) : FunctionACtx(func, tp) {}
  void setIsParallel(BasicBlock* header, bool b) {
    std::cerr << "set " << header->name() << " is parallel " << b << std::endl;
    mLpIsParallel[header] = b;
  }
  bool getIsParallel(BasicBlock* lp) {
    if (mLpIsParallel.count(lp)) {
      return mLpIsParallel[lp];
    }
    assert(false and "input an unexistend loop in ");
  }
  std::set<PhiInst*>& resPhi(BasicBlock* bb) { return mLpPhis[bb]; }
  void clearAll() {
    mLpIsParallel.clear();
    mLpPhis.clear();
  }
  void refresh() {}
  // set
  void setPhi(PhiInst* phi, bool isadd, bool issub, bool ismul, Value* mod) {
    mIsPhiAdd[phi] = isadd;
    mIsPhiMul[phi] = ismul;
    mIsPhiSub[phi] = issub;
    mModuloVal[phi] = mod;
  }
  // get
  bool getIsAdd(PhiInst* phi) {
    assert(false and "can not use!");
    return mIsPhiAdd.at(phi);
  }
  bool getIsSub(PhiInst* phi) {
    assert(false and "can not use!");
    return mIsPhiSub.at(phi);
  }
  bool getIsMul(PhiInst* phi) {
    assert(false and "can not use!");
    return mIsPhiMul.at(phi);
  }
  Value* getMod(PhiInst* phi) {
    assert(false and "can not use!");
    return mModuloVal.at(phi);
  }
};

};  // namespace pass
