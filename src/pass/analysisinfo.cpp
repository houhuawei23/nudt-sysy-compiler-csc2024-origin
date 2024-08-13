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
    mCallGraph = new callGraph(mModule, this);
    mSideEffectInfo = new sideEffectInfo(mModule, this);
    for (auto func : mModule->funcs()) {
        if (func->blocks().empty()) continue;
        mDomTree[func] = new domTree(func, this);
        mPDomTree[func] = new pdomTree(func, this);
        mLoopInfo[func] = new loopInfo(func, this);
        mIndVarInfo[func] = new indVarInfo(func, this);
        mDepInfo[func] = new dependenceInfo(func,this);
        mParallelInfo[func] = new parallelInfo(func,this);
    }
}

void domTree::refresh() {
    using namespace pass;
    domInfoPass dip = domInfoPass();
    dip.run(passUnit,topManager);
    setOn();
}

void pdomTree::refresh() {
    using namespace pass;
    postDomInfoPass pdi = postDomInfoPass();
    pdi.run(passUnit,topManager);
    setOn();
}

void loopInfo::refresh() {
    using namespace pass;
    loopAnalysis la = loopAnalysis();
    loopInfoCheck lic = loopInfoCheck();
    la.run(passUnit,topManager);
    lic.run(passUnit,topManager);
    setOn();
}
void loopInfo::print(std::ostream& os) const {
    os << "Loop Info:\n";
    for (auto loop : _loops) {
        std::cerr << "level: " << _looplevel.at(loop->header()) << std::endl;
        loop->print(os);
    }
    std::cerr << std::endl;
}
// looplevel small to big
std::vector<ir::Loop*> loopInfo::sortedLoops(bool traverse ) {
    auto loops = _loops;
    std::sort(loops.begin(), loops.end(), [&](Loop* lhs, Loop* rhs) {
        return _looplevel.at(lhs->header()) < _looplevel.at(rhs->header());
    });
    return std::move(loops);
}

void callGraph::refresh() {
    using namespace pass;
    callGraphBuild cgb = callGraphBuild();
    cgb.run(passUnit,topManager);

    setOn();
}

void indVarInfo::refresh() {
    using namespace pass;
    // PassManager pm = PassManager(passUnit->module(), topManager);
    indVarAnalysis iva = indVarAnalysis();
    // indVarInfoCheck ivc = indVarInfoCheck();
    iva.run(passUnit,topManager);
    // ivc.run(passUnit,topManager);
    setOn();
}

void sideEffectInfo::refresh() {
    using namespace pass;
    // PassManager pm = PassManager(passUnit, topManager);
    sideEffectAnalysis sea = sideEffectAnalysis();
    sea.run(passUnit,topManager);
    setOn();
}

void dependenceInfo::refresh() {
    using namespace pass;
    dependenceAnalysis da=dependenceAnalysis();
    da.run(passUnit,topManager);
    setOn();
}
