#include "ir/ir.hpp"
#include "pass/analysisinfo.hpp"
#include "pass/analysis/dom.hpp"
#include "pass/analysis/pdom.hpp"
#include "pass/analysis/callgraph.hpp"
#include "pass/analysis/loop.hpp"
#include "pass/analysis/indvar.hpp"

using namespace pass;

void topAnalysisInfoManager::initialize(){
    mCallGraph=new callGraph(mModule,this);
    for(auto func:mModule->funcs()){
        if(func->blocks().empty())continue;
        mDomTree[func]=new domTree(func,this);
        mPDomTree[func]=new pdomTree(func,this);
        mLoopInfo[func]=new loopInfo(func,this);
        mIndVarInfo[func]=new indVarInfo(func,this);
    }
}

void domTree::refresh(){
    if(not _isvalid){
        using namespace pass;
        PassManager pm=PassManager(_pu->module(),_tp);
        domInfoPass dip=domInfoPass();
        pm.run(&dip);
        setOn();
    }
}

void pdomTree::refresh(){
    if(not _isvalid){
        using namespace pass;
        PassManager pm=PassManager(_pu->module(),_tp);
        postDomInfoPass pdi=postDomInfoPass();
        pm.run(&pdi);
        setOn();
    }
}


void loopInfo::refresh(){
    if(not _isvalid){
        using namespace pass;
        PassManager pm=PassManager(_pu->module(),_tp);
        loopAnalysis la=loopAnalysis();
        pm.run(&la);
        setOn();
    }
}


void callGraph::refresh(){
    if(not _isvalid){
        using namespace pass;
        PassManager pm=PassManager(_pu,_tp);
        callGraphBuild cgb=callGraphBuild();
        pm.run(&cgb);
        // callGraphCheck cgc=callGraphCheck();
        // pm.run(&cgc);
        setOn();
    }
}

void indVarInfo::refresh(){
    if(not _isvalid){
        using namespace pass;
        PassManager pm=PassManager(_pu->module(),_tp);
        indVarAnalysis iva=indVarAnalysis();
        pm.run(&iva);
        setOn();
    }
}

void indVarInfo::initialize(){
    auto lpctx=_tp->getLoopInfo(_pu);
    for(auto loop:lpctx->loops()){
        _loopToIndvar[loop]=std::vector<ir::indVar*>();
    }
}

