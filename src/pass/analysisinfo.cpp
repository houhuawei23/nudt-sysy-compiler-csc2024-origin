#include "ir/ir.hpp"
#include "pass/analysisinfo.hpp"
#include "pass/analysis/dom.hpp"
#include "pass/analysis/pdom.hpp"
#include "pass/analysis/callgraph.hpp"
#include "pass/analysis/loop.hpp"
#include "pass/analysis/indvar.hpp"
#include "pass/analysis/sideEffectAnalysis.hpp"
#include "pass/analysis/dependenceAnalysis/DependenceAnalysis.hpp"
#include "pass/analysis/dependenceAnalysis/dpaUtils.hpp"
using namespace ir;
using namespace pass;

void TopAnalysisInfoManager::initialize() {
    mCallGraph = new CallGraph(mModule, this);
    mSideEffectInfo = new SideEffectInfo(mModule, this);
    for (auto func : mModule->funcs()) {
        if (func->blocks().empty()) continue;
        mDomTree[func] = new DomTree(func, this);
        mPDomTree[func] = new PDomTree(func, this);
        mLoopInfo[func] = new LoopInfo(func, this);
        mIndVarInfo[func] = new IndVarInfo(func, this);
        mDepInfo[func] = new dependenceInfo(func,this);
        mParallelInfo[func] = new parallelInfo(func,this);
    }
}

void DomTree::refresh() {
    using namespace pass;
    DomInfoAnalysis dip = DomInfoAnalysis();
    dip.run(passUnit,topManager);
    setOn();
}

void PDomTree::refresh() {
    using namespace pass;
    PostDomInfoPass pdi = PostDomInfoPass();
    pdi.run(passUnit,topManager);
    setOn();
}

void LoopInfo::refresh() {
    using namespace pass;
    LoopAnalysis la = LoopAnalysis();
    // loopInfoCheck lic = loopInfoCheck();
    la.run(passUnit,topManager);
    // lic.run(passUnit,topManager);
    setOn();
}
void LoopInfo::print(std::ostream& os) const {
    os << "Loop Info:\n";
    for (auto loop : _loops) {
        std::cerr << "level: " << _looplevel.at(loop->header()) << std::endl;
        loop->print(os);
    }
    std::cerr << std::endl;
}
// looplevel small to big
std::vector<ir::Loop*> LoopInfo::sortedLoops(bool reverse) {
    auto loops = _loops;
    std::sort(loops.begin(), loops.end(), [&](Loop* lhs, Loop* rhs) {
        return _looplevel.at(lhs->header()) < _looplevel.at(rhs->header());
    });
    if (reverse)
        std::reverse(loops.begin(), loops.end());
    return std::move(loops);
}

void CallGraph::refresh() {
    using namespace pass;
    CallGraphBuild cgb = CallGraphBuild();
    cgb.run(passUnit,topManager);

    setOn();
}

void IndVarInfo::refresh() {
    using namespace pass;
    // PassManager pm = PassManager(passUnit->module(), topManager);
    IndVarAnalysis iva = IndVarAnalysis();
    // indVarInfoCheck ivc = indVarInfoCheck();
    iva.run(passUnit,topManager);
    // ivc.run(passUnit,topManager);
    setOn();
}

void SideEffectInfo::refresh() {
    using namespace pass;
    // PassManager pm = PassManager(passUnit, topManager);
    SideEffectAnalysis sea = SideEffectAnalysis();
    sea.run(passUnit,topManager);
    setOn();
}

void dependenceInfo::refresh() {
    using namespace pass;
    DependenceAnalysis da=DependenceAnalysis();
    da.run(passUnit,topManager);
    setOn();
}
