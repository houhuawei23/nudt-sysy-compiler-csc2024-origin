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
        mDepInfo[func] = new dependenceInfoForLoops(func, this);
    }
}

void domTree::refresh() {
    if (not isValid) {
        using namespace pass;
        domInfoPass dip = domInfoPass();
        dip.run(passUnit,topManager);
        setOn();
    }
}

void pdomTree::refresh() {
    if (not isValid) {
        using namespace pass;
        postDomInfoPass pdi = postDomInfoPass();
        pdi.run(passUnit,topManager);
        setOn();
    }
}

void loopInfo::refresh() {
    if (not isValid) {
        using namespace pass;
        loopAnalysis la = loopAnalysis();
        // loopInfoCheck lic = loopInfoCheck();
        la.run(passUnit,topManager);
        // lic.run(passUnit,topManager);
        setOn();
    }
}

void callGraph::refresh() {
    if (not isValid) {
        using namespace pass;
        callGraphBuild cgb = callGraphBuild();
        cgb.run(passUnit,topManager);

        setOn();
    }
}

void indVarInfo::refresh() {
    if (not isValid) {
        using namespace pass;
        // PassManager pm = PassManager(passUnit->module(), topManager);
        indVarAnalysis iva = indVarAnalysis();
        // indVarInfoCheck ivc = indVarInfoCheck();
        iva.run(passUnit,topManager);
        // ivc.run(passUnit,topManager);
        setOn();
    }
}

void sideEffectInfo::refresh() {
    if(not isValid){
        using namespace pass;
        // PassManager pm = PassManager(passUnit, topManager);
        sideEffectAnalysis sea = sideEffectAnalysis();
        sea.run(passUnit,topManager);
        setOn();
    }
}

void dependenceInfoForLoops::refresh() {
    if(not isValid){
        using namespace pass;
        // PassManager pm = PassManager(passUnit, topManager);
        dependenceAnalysis da=dependenceAnalysis();
        da.run(passUnit,topManager);
        setOn();
    }
}

void dependenceInfoForLoop::print(std::ostream& os){
    using namespace std;
    os<<"loop whose header is \""
        <<lp->header()->name()<<"\", depinfo:"<<std::endl;
    os<<"Used base addrs:"<<endl;
    for(auto bd:baseAddrs){
        auto typebaseaddr=getBaseaddrType(bd);
        if(typebaseaddr==baseAddrType::typeglobal){
            os<<"global:\t";
        }
        if(typebaseaddr==baseAddrType::typelocal){
            os<<"local:\t";
        }
        if(typebaseaddr==baseAddrType::typearg){
            os<<"arg:\t";
        }
        os<<bd->name()<<endl;
        os<<"We got "<<(baseAddrToSubAddrs[bd]).size()<<" sub addrs."<<endl;
    }
    
}