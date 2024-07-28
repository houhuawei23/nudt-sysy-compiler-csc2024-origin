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
    std::unordered_map<ir::BasicBlock*, std::vector<ir::BasicBlock*>>
      _domfrontier;
    std::vector<ir::BasicBlock*> _BFSDomTreeVector;

  public:
    domTree(ir::Function* func, TopAnalysisInfoManager* tp)
      : FunctionACtx(func, tp) {}
    ir::BasicBlock* idom(ir::BasicBlock* bb) { return _idom[bb]; }
    void set_idom(ir::BasicBlock* bb, ir::BasicBlock* idbb) {
        _idom[bb] = idbb;
    }
    ir::BasicBlock* sdom(ir::BasicBlock* bb) { return _sdom[bb]; }
    void set_sdom(ir::BasicBlock* bb, ir::BasicBlock* sdbb) {
        _sdom[bb] = sdbb;
    }
    int domlevel(ir::BasicBlock* bb) { return _domlevel[bb]; }
    void set_domlevel(ir::BasicBlock* bb, int lv) { _domlevel[bb] = lv; }

    std::vector<ir::BasicBlock*>& domson(ir::BasicBlock* bb) {
        return _domson[bb];
    }

    std::vector<ir::BasicBlock*>& domfrontier(ir::BasicBlock* bb) {
        return _domfrontier[bb];
    }

    std::vector<ir::BasicBlock*>& BFSDomTreeVector() {
        return _BFSDomTreeVector;
    }
    void clearAll() {
        _idom.clear();
        _sdom.clear();
        _domson.clear();
        _domfrontier.clear();
        _domlevel.clear();
        _BFSDomTreeVector.clear();
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
        if (bb1 == bb2) return true;
        auto bbIdom = _idom[bb2];
        while (bbIdom != nullptr) {
            if (bbIdom == bb1) return true;
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
};

class pdomTree : public FunctionACtx {  // also used as pdom
  protected:
    std::unordered_map<ir::BasicBlock*, ir::BasicBlock*> _ipdom;
    std::unordered_map<ir::BasicBlock*, ir::BasicBlock*> _spdom;
    std::unordered_map<ir::BasicBlock*, int> _pdomlevel;
    std::unordered_map<ir::BasicBlock*, std::vector<ir::BasicBlock*>> _pdomson;
    std::unordered_map<ir::BasicBlock*, std::vector<ir::BasicBlock*>>
      _pdomfrontier;

  public:
    pdomTree(ir::Function* func, TopAnalysisInfoManager* tp)
      : FunctionACtx(func, tp) {}
    ir::BasicBlock* ipdom(ir::BasicBlock* bb) { return _ipdom[bb]; }
    void set_ipdom(ir::BasicBlock* bb, ir::BasicBlock* idbb) {
        _ipdom[bb] = idbb;
    }
    ir::BasicBlock* spdom(ir::BasicBlock* bb) { return _spdom[bb]; }
    void set_spdom(ir::BasicBlock* bb, ir::BasicBlock* sdbb) {
        _spdom[bb] = sdbb;
    }
    int pdomlevel(ir::BasicBlock* bb) { return _pdomlevel[bb]; }
    void set_pdomlevel(ir::BasicBlock* bb, int lv) { _pdomlevel[bb] = lv; }
    std::vector<ir::BasicBlock*>& pdomson(ir::BasicBlock* bb) {
        return _pdomson[bb];
    }
    std::vector<ir::BasicBlock*>& pdomfrontier(ir::BasicBlock* bb) {
        return _pdomfrontier[bb];
    }
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
    void refresh() override;
};

class loopInfo : public FunctionACtx {
  protected:
    std::vector<ir::Loop*> _loops;
    std::unordered_map<ir::BasicBlock*, ir::Loop*> _head2loop;
    std::unordered_map<ir::BasicBlock*, size_t> _looplevel;

  public:
    loopInfo(ir::Function* fp, TopAnalysisInfoManager* tp)
      : FunctionACtx(fp, tp) {}
    std::vector<ir::Loop*>& loops() { return _loops; }
    ir::Loop* head2loop(ir::BasicBlock* bb) {
        if (_head2loop.count(bb) == 0) return nullptr;
        return _head2loop[bb];
    }
    void set_head2loop(ir::BasicBlock* bb, ir::Loop* lp) {
        _head2loop[bb] = lp;
    }
    int looplevel(ir::BasicBlock* bb) { return _looplevel[bb]; }
    void set_looplevel(ir::BasicBlock* bb, int lv) { _looplevel[bb] = lv; }
    void clearAll() {
        _loops.clear();
        _head2loop.clear();
    }
    bool isHeader(ir::BasicBlock* bb) { return _head2loop.count(bb); }
    ir::Loop* getinnermostLoop(ir::BasicBlock* bb) {  // 返回最内层的循环
        ir::Loop* innermost = nullptr;
        for (auto L : _loops) {
            if (L->contains(bb)) {
                if (innermost == nullptr)
                    innermost = L;
                else {
                    if (_looplevel[L->header()] <
                        _looplevel[innermost->header()])
                        innermost = L;
                }
            }
        }
        return innermost;
    }
    void refresh() override;
};

class callGraph : public ModuleACtx {
  protected:
    std::unordered_map<ir::Function*, std::set<ir::Function*>> _callees;
    std::unordered_map<ir::Function*, std::set<ir::Function*>> _callers;
    std::unordered_map<ir::Function*, bool> _is_called;
    std::unordered_map<ir::Function*, bool> _is_inline;
    std::unordered_map<ir::Function*, bool> _is_lib;

  public:
    callGraph(ir::Module* md, TopAnalysisInfoManager* tp)
      : ModuleACtx(md, tp) {}
    std::set<ir::Function*>& callees(ir::Function* func) {
        return _callees[func];
    }
    std::set<ir::Function*>& callers(ir::Function* func) {
        return _callers[func];
    }
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
    bool isNoCallee(ir::Function* func) { return _callees[func].empty(); }
    void refresh() override;
};

class indVarInfo : public FunctionACtx {
  private:
    std::unordered_map<ir::Loop*, ir::indVar*> _loopToIndvar;

  public:
    indVarInfo(ir::Function* fp, TopAnalysisInfoManager* tp)
      : FunctionACtx(fp, tp) {}
    ir::indVar* getIndvar(ir::Loop* loop) {
        if (_loopToIndvar.count(loop) == 0) return nullptr;
        return _loopToIndvar[loop];
    }
    void clearAll() { _loopToIndvar.clear(); }
    void refresh() override;
    void addIndVar(ir::Loop* lp, ir::indVar* idv) { _loopToIndvar[lp] = idv; }
};

class sideEffectInfo : public ModuleACtx {
  private:
    std::unordered_map<ir::Function*,bool> _hasSideEffect;

  public:
    sideEffectInfo(ir::Module* ctx, TopAnalysisInfoManager* tp)
      : ModuleACtx(ctx, tp) {}
    void clearAll() { _hasSideEffect.clear();  }
    void refresh() override;
    void setFunc(ir::Function* func,bool b){_hasSideEffect[func]=b;}
    bool hasSideEffect(ir::Function* func){return _hasSideEffect[func];}

};
};  // namespace pass
