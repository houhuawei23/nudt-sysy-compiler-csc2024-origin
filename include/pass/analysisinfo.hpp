#pragma once
// add analysis passes
#include "ir/ir.hpp"
#include <unordered_map>
#include <vector>
#include <queue>

namespace pass {
template <typename PassUnit>
class analysisInfo;

class domTree;
class pdomTree;
class loopInfo;
class callGraph;
class indVarInfo;
class TopAnalysisInfoManager;
class dependenceAnalysis;
class LoopDependenceInfo;


template <typename PassUnit>
class analysisInfo {
  protected:
    PassUnit* passUnit;
    TopAnalysisInfoManager* topManager;
    bool isValid;

  public:
    analysisInfo(PassUnit* mp, TopAnalysisInfoManager* mtp, bool v = false)
      : isValid(v), passUnit(mp), topManager(mtp) {}
    void setOn() { isValid = true; }
    void setOff() { isValid = false; }
    virtual void refresh() = 0;
};
using ModuleACtx = analysisInfo<ir::Module>;
using FunctionACtx = analysisInfo<ir::Function>;

// add new analysis info of ir here!
// dom Tree
class domTree : public FunctionACtx {
  protected:
    std::unordered_map<ir::BasicBlock*, ir::BasicBlock*> _idom;
    std::unordered_map<ir::BasicBlock*, ir::BasicBlock*> _sdom;
    std::unordered_map<ir::BasicBlock*, int> _domlevel;
    std::unordered_map<ir::BasicBlock*, std::vector<ir::BasicBlock*>> _domson;
    std::unordered_map<ir::BasicBlock*, std::vector<ir::BasicBlock*>> _domfrontier;
    std::vector<ir::BasicBlock*> _BFSDomTreeVector;
    std::vector<ir::BasicBlock*> _DFSDomTreeVector;

  public:
    domTree(ir::Function* func, TopAnalysisInfoManager* tp) : FunctionACtx(func, tp) {}
    ir::BasicBlock* idom(ir::BasicBlock* bb) { return _idom[bb]; }
    void set_idom(ir::BasicBlock* bb, ir::BasicBlock* idbb) { _idom[bb] = idbb; }
    ir::BasicBlock* sdom(ir::BasicBlock* bb) { return _sdom[bb]; }
    void set_sdom(ir::BasicBlock* bb, ir::BasicBlock* sdbb) { _sdom[bb] = sdbb; }
    int domlevel(ir::BasicBlock* bb) { return _domlevel[bb]; }
    void set_domlevel(ir::BasicBlock* bb, int lv) { _domlevel[bb] = lv; }

    std::vector<ir::BasicBlock*>& domson(ir::BasicBlock* bb) { return _domson[bb]; }

    std::vector<ir::BasicBlock*>& domfrontier(ir::BasicBlock* bb) { return _domfrontier[bb]; }

    std::vector<ir::BasicBlock*>& BFSDomTreeVector() { return _BFSDomTreeVector; }

    std::vector<ir::BasicBlock*>& DFSDomTreeVector() { return _DFSDomTreeVector; }

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
            _domson[bb] = std::vector<ir::BasicBlock*>();
            _domfrontier[bb] = std::vector<ir::BasicBlock*>();
        }
    }
    void refresh() override;
    bool dominate(ir::BasicBlock* bb1, ir::BasicBlock* bb2) {
        if (bb1 == bb2)
            return true;
        auto bbIdom = _idom[bb2];
        while (bbIdom != nullptr) {
            if (bbIdom == bb1)
                return true;
            bbIdom = _idom[bbIdom];
        }
        return false;
    }
    void BFSDomTreeInfoRefresh() {
        std::queue<ir::BasicBlock*> bbqueue;
        std::unordered_map<ir::BasicBlock*, bool> vis;
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
        std::stack<ir::BasicBlock*> bbstack;
        std::unordered_map<ir::BasicBlock*, bool> vis;
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

class pdomTree : public FunctionACtx {  // also used as pdom
  protected:
    std::unordered_map<ir::BasicBlock*, ir::BasicBlock*> _ipdom;
    std::unordered_map<ir::BasicBlock*, ir::BasicBlock*> _spdom;
    std::unordered_map<ir::BasicBlock*, int> _pdomlevel;
    std::unordered_map<ir::BasicBlock*, std::vector<ir::BasicBlock*>> _pdomson;
    std::unordered_map<ir::BasicBlock*, std::vector<ir::BasicBlock*>> _pdomfrontier;

  public:
    pdomTree(ir::Function* func, TopAnalysisInfoManager* tp) : FunctionACtx(func, tp) {}
    ir::BasicBlock* ipdom(ir::BasicBlock* bb) {
        assert(bb && "bb is null");
        return _ipdom[bb];
    }
    void set_ipdom(ir::BasicBlock* bb, ir::BasicBlock* idbb) {
        _ipdom[bb] = idbb;
    }
    ir::BasicBlock* spdom(ir::BasicBlock* bb) {
        assert(bb && "bb is null");
        return _spdom[bb];
    }
    void set_spdom(ir::BasicBlock* bb, ir::BasicBlock* sdbb) {
        _spdom[bb] = sdbb;
    }
    int pdomlevel(ir::BasicBlock* bb) {
        assert(bb && "bb is null");
        return _pdomlevel[bb];
    }
    void set_pdomlevel(ir::BasicBlock* bb, int lv) { _pdomlevel[bb] = lv; }
    std::vector<ir::BasicBlock*>& pdomson(ir::BasicBlock* bb) { return _pdomson[bb]; }
    std::vector<ir::BasicBlock*>& pdomfrontier(ir::BasicBlock* bb) { return _pdomfrontier[bb]; }
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
            _pdomson[bb] = std::vector<ir::BasicBlock*>();
            _pdomfrontier[bb] = std::vector<ir::BasicBlock*>();
        }
    }

    bool pdominate(ir::BasicBlock* bb1, ir::BasicBlock* bb2) {
        if (bb1 == bb2)
            return true;
        auto bbIdom = _ipdom[bb2];
        while (bbIdom != nullptr) {
            if (bbIdom == bb1)
                return true;
            bbIdom = _ipdom[bbIdom];
        }
        return false;
    }

    void refresh() override;
};

class loopInfo : public FunctionACtx {
  protected:
    std::vector<ir::Loop*> _loops;
    std::unordered_map<ir::BasicBlock*, ir::Loop*> _head2loop;
    std::unordered_map<ir::BasicBlock*, size_t> _looplevel;

  public:
    loopInfo(ir::Function* fp, TopAnalysisInfoManager* tp) : FunctionACtx(fp, tp) {}
    std::vector<ir::Loop*>& loops() { return _loops; }
    ir::Loop* head2loop(ir::BasicBlock* bb) {
        if (_head2loop.count(bb) == 0)
            return nullptr;
        return _head2loop[bb];
    }
    void set_head2loop(ir::BasicBlock* bb, ir::Loop* lp) { _head2loop[bb] = lp; }
    int looplevel(ir::BasicBlock* bb) { return _looplevel[bb]; }
    void set_looplevel(ir::BasicBlock* bb, int lv) { _looplevel[bb] = lv; }
    void clearAll() {
        _loops.clear();
        _head2loop.clear();
        _looplevel.clear();
    }
    bool isHeader(ir::BasicBlock* bb) { return _head2loop.count(bb); }
    ir::Loop* getinnermostLoop(ir::BasicBlock* bb) {  // 返回最内层的循环
        ir::Loop* innermost = nullptr;
        for (auto L : _loops) {
            if (L->contains(bb)) {
                if (innermost == nullptr)
                    innermost = L;
                else {
                    if (_looplevel[L->header()] < _looplevel[innermost->header()])
                        innermost = L;
                }
            }
        }
        return innermost;
    }
    void refresh() override;
    void print(std::ostream& os) const;
    std::vector<ir::Loop*> sortedLoops(bool traverse = false); // looplevel small to big
};

class callGraph : public ModuleACtx {
  protected:
    std::unordered_map<ir::Function*, std::set<ir::Function*>> _callees;
    std::unordered_map<ir::Function*, std::set<ir::Function*>> _callers;
    std::unordered_map<ir::Function*, bool> _is_called;
    std::unordered_map<ir::Function*, bool> _is_inline;
    std::unordered_map<ir::Function*, bool> _is_lib;
    std::unordered_map<ir::Function*, std::set<ir::CallInst*>>_callerCallInsts;//func's caller insts are func's callers'
    std::unordered_map<ir::Function*, std::set<ir::CallInst*>>_calleeCallInsts;//func's callee insts are func's

  public:
    callGraph(ir::Module* md, TopAnalysisInfoManager* tp) : ModuleACtx(md, tp) {}
    std::set<ir::Function*>& callees(ir::Function* func) { return _callees[func]; }
    std::set<ir::Function*>& callers(ir::Function* func) { return _callers[func]; }
    std::set<ir::CallInst*>& callerCallInsts(ir::Function* func) { return _callerCallInsts[func]; }
    std::set<ir::CallInst*>& calleeCallInsts(ir::Function* func) { return _calleeCallInsts[func]; }
    bool isCalled(ir::Function* func) { return _is_called[func]; }
    bool isInline(ir::Function* func) { return _is_inline[func]; }
    bool isLib(ir::Function* func) { return _is_lib[func]; }
    void set_isCalled(ir::Function* func, bool b) { _is_called[func] = b; }
    void set_isInline(ir::Function* func, bool b) { _is_inline[func] = b; }
    void set_isLib(ir::Function* func, bool b) { _is_lib[func] = b; }
    void clearAll() {
        _callees.clear();
        _callers.clear();
        _is_called.clear();
        _is_inline.clear();
        _is_lib.clear();
    }
    void initialize() {
        for (auto func : passUnit->funcs()) {
            _callees[func] = std::set<ir::Function*>();
            _callers[func] = std::set<ir::Function*>();
        }
    }
    bool isNoCallee(ir::Function* func) {
        if (_callees[func].size() == 0)
            return true;
        for (auto f : _callees[func]) {
            if (not isLib(f))
                return false;
        }
        return true;
    }
    void refresh() override;
};

class indVarInfo : public FunctionACtx {
  private:
    std::unordered_map<ir::Loop*, ir::IndVar*> _loopToIndvar;

  public:
    indVarInfo(ir::Function* fp, TopAnalysisInfoManager* tp) : FunctionACtx(fp, tp) {}
    ir::IndVar* getIndvar(ir::Loop* loop) {
        if (_loopToIndvar.count(loop) == 0)
            return nullptr;
        return _loopToIndvar[loop];
    }
    void clearAll() { _loopToIndvar.clear(); }
    void refresh() override;
    void addIndVar(ir::Loop* lp, ir::IndVar* idv) { _loopToIndvar[lp] = idv; }
};

class sideEffectInfo : public ModuleACtx {
  private:
    std::unordered_map<ir::Function*, std::set<ir::GlobalVariable*>> _FuncReadGlobals;//当前函数读取的全局变量
    std::unordered_map<ir::Function*, std::set<ir::GlobalVariable*>> _FuncWriteGlobals;//当前函数写入的全局变量
    std::unordered_map<ir::Argument*,bool> _isArgumentRead;//对于当前argument函数是否读取（仅限pointer）
    std::unordered_map<ir::Argument*,bool> _isArgumentWrite;//对于当前argument哈数是否写入（仅限pointer）
    std::unordered_map<ir::Function*,bool> _isLib;//当前函数是否为lib函数
    std::unordered_map<ir::Function*,std::set<ir::Argument*>> _funcPointerArgs; //当前函数的参数中有哪些是指针参数
    std::unordered_map<ir::Function*,bool> _isCallLibFunc;//当前函数有无调用库函数或者简介调用库函数
    

  public:
    sideEffectInfo(ir::Module* ctx, TopAnalysisInfoManager* tp) : ModuleACtx(ctx, tp) {}
    void clearAll() {
        _FuncReadGlobals.clear();
        _FuncWriteGlobals.clear();
        _isArgumentRead.clear();
        _isArgumentWrite.clear();
        _isLib.clear();
        _funcPointerArgs.clear();
        _isCallLibFunc.clear();
    }
    void refresh() override;
    //get
    bool getArgRead(ir::Argument* arg){return _isArgumentRead[arg];}
    bool getArgWrite(ir::Argument* arg){return _isArgumentWrite[arg];}
    bool getIsLIb(ir::Function* func){return _isLib[func];}
    bool getIsCallLib(ir::Function* func){return _isCallLibFunc[func];}
    //set
    void setArgRead(ir::Argument* arg,bool b){_isArgumentRead[arg]=b;}
    void setArgWrite(ir::Argument* arg,bool b){_isArgumentWrite[arg]=b;}
    void setFuncIsLIb(ir::Function* func,bool b){_isLib[func]=b;}
    void setFuncIsCallLib(ir::Function* func,bool b){_isCallLibFunc[func]=b;}
    //reference
    std::set<ir::GlobalVariable*>& funcReadGlobals(ir::Function* func){return _FuncReadGlobals[func];}
    std::set<ir::GlobalVariable*>& funcWriteGlobals(ir::Function* func){return _FuncWriteGlobals[func];}
    std::set<ir::Argument*>& funcArgSet(ir::Function* func){return _funcPointerArgs[func];}
    //old API
    bool hasSideEffect(ir::Function* func){
        if(_isLib[func])return true;
        if(_isCallLibFunc[func])return true;
        if(not _FuncWriteGlobals[func].empty())return true;
        
        for(auto arg:_funcPointerArgs[func]){
            if(getArgWrite(arg))return true;
        }
        return false;
    }
    bool isPureFunc(ir::Function* func){
        for(auto arg:funcArgSet(func)){
            if(getArgRead(arg))return false;
        }
        return (not hasSideEffect(func) and _FuncReadGlobals[func].empty()) and not _isLib[func];
    }
    void functionInit(ir::Function* func){
        _FuncReadGlobals[func]=std::set<ir::GlobalVariable*>();
        _FuncWriteGlobals[func]=std::set<ir::GlobalVariable*>();
        for(auto arg:func->args()){
            if(not arg->isPointer())continue;
            _funcPointerArgs[func].insert(arg);
            setArgRead(arg,false);
            setArgWrite(arg,false);
        }
        _isCallLibFunc[func]=false;
    }
};

class dependenceInfo:public FunctionACtx{
    private:
        std::unordered_map<ir::Loop*,LoopDependenceInfo*>funcToLoopDependenceInfo;
    public:
        dependenceInfo(ir::Function* func, TopAnalysisInfoManager* tp) : FunctionACtx(func, tp) {}
        LoopDependenceInfo* getLoopDependenceInfo(ir::Loop* lp){
            if(funcToLoopDependenceInfo.count(lp))
                return funcToLoopDependenceInfo[lp];
            else return nullptr;
        }
        void clearAll(){funcToLoopDependenceInfo.clear();}
        void refresh() override;
        void setDepInfoLp(ir::Loop* lp,LoopDependenceInfo* input){
            funcToLoopDependenceInfo[lp]=input;
        }
};

class parallelInfo:public FunctionACtx{
    //你想并行的这里都有!
    private:
        std::unordered_map<ir::BasicBlock*,bool>_LpIsParallel;
        std::unordered_map<ir::BasicBlock*,std::set<ir::PhiInst*>>_LpPhis;
    public:
        parallelInfo(ir::Function* func,TopAnalysisInfoManager* tp) : FunctionACtx(func,tp) {}
        void setIsParallel(ir::BasicBlock* lp,bool b){_LpIsParallel[lp]=b;}
        bool getIsParallel(ir::BasicBlock* lp){
            if(_LpIsParallel.count(lp)){
                return _LpIsParallel[lp];
            }
            assert(false and "input an unexistend loop in ");
        }
        std::set<ir::PhiInst*>& resPhi(ir::BasicBlock* bb){return _LpPhis[bb];}
        void clearAll(){
            _LpIsParallel.clear();
            _LpPhis.clear();
        }
        void refresh(){}

};

};  // namespace pass
