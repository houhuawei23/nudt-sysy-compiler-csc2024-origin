#include "pass/analysis/loop.hpp"
#include <unordered_map>

static std::unordered_map<ir::BasicBlock*,int>stLoopLevel;

namespace pass{
    void loopAnalysis::run(ir::Function* func,topAnalysisInfoManager* tp){
        if(func->isOnlyDeclare())return;
        domctx=tp->getDomTree(func);
        domctx->refresh();
        lpctx=tp->getLoopInfo(func);
        for(auto bb:func->blocks())lpctx->set_looplevel(bb,0);
        for(auto bb:func->blocks()){
            if(bb->pre_blocks().empty())continue;
            for(auto bbPre:bb->pre_blocks()){
                if(domctx->dominate(bb,bbPre)){//bb->dominate(bbPre)
                    addLoopBlocks(func,bb,bbPre);
                }
            }
        }
        for(auto iLoop:lpctx->loops()){
            loopGetExits(iLoop);
        }
    }
    void loopAnalysis::addLoopBlocks(ir::Function*func,ir::BasicBlock* header,ir::BasicBlock* tail){
        ir::Loop* curLoop;
        //auto &headerToLoop=func->headToLoop();
        //auto findLoop=headerToLoop.find(header);
        auto findLoop=lpctx->head2loop(header);
        if(findLoop==nullptr){
            curLoop=new ir::Loop(header,func);
            curLoop->latchs().insert(tail);
            // func->Loops().push_back(curLoop);
            lpctx->loops().push_back(curLoop);
            // headerToLoop[header]=curLoop;
            lpctx->set_head2loop(header,curLoop);
            // header->looplevel++;
            lpctx->set_looplevel(header,lpctx->looplevel(header)+1);
        }
        else{
            // curLoop=headerToLoop[header];
            curLoop=lpctx->head2loop(header);
            curLoop->latchs().insert(tail);
        }
        ir::block_ptr_stack bbStack;
        bbStack.push(tail);
        while(not bbStack.empty()){
            auto curBB=bbStack.top();
            bbStack.pop();
            curLoop->blocks().insert(curBB);
            // curBB->looplevel++;
            lpctx->set_looplevel(curBB,(lpctx->looplevel(curBB))+1);
            for(auto curBBPre:curBB->pre_blocks()){
                if(curBBPre==header)continue;
                if(curLoop->blocks().find(curBBPre)==curLoop->blocks().end()){
                    bbStack.push(curBBPre);
                }
            }
        }
    }

    void loopAnalysis::loopGetExits(ir::Loop* plp){
        for(auto bb:plp->blocks()){
            for(auto bbNext:bb->next_blocks()){
                if(plp->blocks().count(bbNext)==0){
                    plp->exits().insert(bbNext);
                }
            }
        }
    }

    void loopInfoCheck::run(ir::Function* func,topAnalysisInfoManager* tp){
        using namespace std;
        if(func->isOnlyDeclare())return;
        lpctx=tp->getLoopInfo(func);
        cout<<"In Function \""<<func->name()<<"\": "<<endl;
        int cnt=0;
        for(auto loop:lpctx->loops()){//func->Loops()
            cout<<"Loop "<<cnt<<":"<<endl;
            cout<<"Header: "<<loop->header()->name()<<endl;
            cout<<"Loop Blocks: "<<endl;
            for(auto bb:loop->blocks()){
                cout<<bb->name()<<"\t";
            }
            cout<<endl<<endl;
            cnt++;
        }
        cout<<"Loop Level:"<<endl;
        for(auto bb:func->blocks()){
            cout<<bb->name()<<" : "<<lpctx->looplevel(bb)<<endl;
        }
    }
}