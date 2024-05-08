#include "pass/analysis/loop.hpp"

namespace pass{
    std::string loopAnalysis::name(){return "loopAnalysis";}
    void loopAnalysis::run(ir::Function* func){
        if(!func->entry())return;
        for(auto bb:func->blocks())bb->looplevel=0;
        for(auto bb:func->blocks()){
            if(bb->pre_blocks().empty())continue;
            for(auto bbPre:bb->pre_blocks()){
                if(bb->dominate(bbPre)){
                    addLoopBlocks(func,bb,bbPre);
                }
            }
        }
    }
    void loopAnalysis::addLoopBlocks(ir::Function*func,ir::BasicBlock* header,ir::BasicBlock* tail){
        ir::Loop* curLoop;
        auto &headerToLoop=func->headToLoop();
        auto findLoop=headerToLoop.find(header);
        if(findLoop==headerToLoop.end()){
            curLoop=new ir::Loop(header,func);
            func->Loops().push_back(curLoop);
            headerToLoop[header]=curLoop;
            header->looplevel++;
        }
        else{
            curLoop=headerToLoop[header];
        }
        ir::block_ptr_stack bbStack;
        bbStack.push(tail);
        while(not bbStack.empty()){
            auto curBB=bbStack.top();
            bbStack.pop();
            curLoop->blocks().insert(curBB);
            curBB->looplevel++;
            for(auto curBBPre:curBB->pre_blocks()){
                if(curBBPre==header)continue;
                if(curLoop->blocks().find(curBBPre)==curLoop->blocks().end()){
                    bbStack.push(curBBPre);
                }
            }
        }
    }
    std::string loopInfoCheck::name(){return "loopInfoCheck";}
    void loopInfoCheck::run(ir::Function* func){
        using namespace std;
        if(!func->entry())return;
        cout<<"In Function \""<<func->name()<<"\": "<<endl;
        int cnt=0;
        for(auto loop:func->Loops()){
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
            cout<<bb->name()<<" : "<<bb->looplevel<<endl;
        }
    }
}