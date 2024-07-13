#include "ir/ir.hpp"
#include "pass/analysisinfo.hpp"
#include "pass/analysis/dom.hpp"
#include "pass/analysis/pdom.hpp"
#include "pass/analysis/callgraph.hpp"
#include "pass/analysis/loop.hpp"

using namespace pass;

void topAnalysisInfoManager::initialize(){
    mCallGraph=new callGraph(mModule,this);
    for(auto func:mModule->funcs()){
        if(func->blocks().empty())continue;
        mDomTree[func]=new domTree(func,this);
        mPDomTree[func]=new pdomTree(func,this);
        mLoopInfo[func]=new loopInfo(func,this);
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


