#include "ir/ir.hpp"
#include "pass/analysisinfo.hpp"
#include "pass/analysis/dom.hpp"
#include "pass/analysis/pdom.hpp"
#include "pass/analysis/callgraph.hpp"
#include "pass/analysis/loop.hpp"
#include "pass/analysis/indvar.hpp"
#include "pass/analysis/sideEffectAnalysis.hpp"

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
    }
}

void domTree::refresh() {
    if (not isValid) {
        using namespace pass;
        PassManager pm = PassManager(passUnit->module(), topManager);
        domInfoPass dip = domInfoPass();
        pm.run(&dip);
        setOn();
    }
}

void pdomTree::refresh() {
    if (not isValid) {
        using namespace pass;
        PassManager pm = PassManager(passUnit->module(), topManager);
        postDomInfoPass pdi = postDomInfoPass();
        pm.run(&pdi);
        setOn();
    }
}

void loopInfo::refresh() {
    if (not isValid) {
        using namespace pass;
        PassManager pm = PassManager(passUnit->module(), topManager);
        loopAnalysis la = loopAnalysis();
        pm.run(&la);
        setOn();
    }
}

void callGraph::refresh() {
    if (not isValid) {
        using namespace pass;
        PassManager pm = PassManager(passUnit, topManager);
        callGraphBuild cgb = callGraphBuild();
        pm.run(&cgb);
        // callGraphCheck cgc=callGraphCheck();
        // pm.run(&cgc);
        setOn();
    }
}

void indVarInfo::refresh() {
    if (not isValid) {
        using namespace pass;
        PassManager pm = PassManager(passUnit->module(), topManager);
        indVarAnalysis iva = indVarAnalysis();
        indVarInfoCheck ivc = indVarInfoCheck();
        pm.run(&iva);
        pm.run(&ivc);
        setOn();
    }
}

void sideEffectInfo::refresh() {
    if(not isValid){
        using namespace pass;
        PassManager pm = PassManager(passUnit, topManager);
        sideEffectAnalysis sea = sideEffectAnalysis();
        pm.run(&sea);
        setOn();
    }
}