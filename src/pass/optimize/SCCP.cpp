#include "pass/optimize/SCCP.hpp"


std::deque<ir::Instruction*>worklist;
std::unordered_map<ir::BasicBlock*,std::unordered_map<ir::BasicBlock*,bool>>executableFlag;
std::unordered_map<ir::BasicBlock*,bool>deadFlag;
std::unordered_map<ir::BasicBlock*,int>livePreNum;


namespace pass{
    std::string SCCP::name(){return "SCCP";}

    bool SCCP::getExecutableFlag(ir::BasicBlock* a, ir::BasicBlock* b){//一开始每一条边都是可执行
        if(executableFlag.find(a)!=executableFlag.end()){
            if(executableFlag[a].find(b)!=executableFlag[a].end()){
                return executableFlag[a][b];
            }
        }
        executableFlag[a][b]=true;
        return true;
    }

    bool SCCP::getDeadFlag(ir::BasicBlock* a){//一开始每一个块都不是死的
        if(deadFlag.find(a)!=deadFlag.end())
            return deadFlag[a];
        else{
            deadFlag[a]=false;
            return deadFlag[a];
        }
    }

    void SCCP::run(ir::Function* func){
        if(!func->entry())return;
        livePreNum.clear();
        worklist.clear();
        executableFlag.clear();
        deadFlag.clear();
        for(auto bb:func->blocks()){
            livePreNum[bb]=bb->pre_blocks().size();
        }
        // func->print(std::cout);
        for(auto bb:func->blocks()){
            for(auto inst:bb->insts()){
                if(inst->getConstantRepl()){//能够进行常数传播
                    worklist.push_back(inst);
                    continue;
                }
                auto brInst=dyn_cast<ir::BranchInst>(inst);
                if(brInst and brInst->is_cond()){//是常数的条件分支
                    auto constCond=dyn_cast<ir::Constant>(brInst->cond());
                    if(constCond)
                        worklist.push_back(inst);
                }
            }
        }

        while(not worklist.empty()){
            auto curinst=worklist.front();
            worklist.pop_front();
            if(getDeadFlag(curinst->parent()))//当前这个语句在死块,直接continue
                continue;
            if(curinst->getConstantRepl()){
                // func->print(std::cout);
                addConstFlod(curinst);
            }
            else{
                auto brInst=dyn_cast<ir::BranchInst>(curinst);
                branchInstFlod(brInst);
            }
        }
        for(auto bbIter=func->blocks().begin();bbIter!=func->blocks().end();){
            auto bb=*bbIter;
            bbIter++;
            if(deadFlag[bb])
                func->force_delete_block(bb);
        }

        // func->print(std::cout);
    }

    void SCCP::addConstFlod(ir::Instruction*inst){
        // inst->parent()->parent()->print(std::cout);
        if(getDeadFlag(inst->parent()))return;
        auto replValue=inst->getConstantRepl();
        for(auto puse:inst->uses()){
            auto useInst=dyn_cast<ir::Instruction>(puse->user());
            assert(useInst);
            useInst->set_operand(puse->index(),replValue);//进行常数替换
            if(useInst->getConstantRepl())//被替换的能不能再传播
                worklist.push_back(useInst);
            else{//分支
                auto brInst=dyn_cast<ir::BranchInst>(useInst);
                if(brInst and brInst->is_cond() and dyn_cast<ir::Constant>(brInst->cond()))
                    worklist.push_back(useInst);
            }
        }
        inst->uses().clear();
        inst->parent()->delete_inst(inst);
    }

    void SCCP::branchInstFlod(ir::BranchInst* brInst){
        assert(brInst->is_cond());
        assert(dyn_cast<ir::Constant>(brInst->cond()));
        if(getDeadFlag(brInst->parent()))return;
        auto constCond=dyn_cast<ir::Constant>(brInst->cond())->i1();//常数的cond
        auto bbdel=constCond?brInst->iffalse():brInst->iftrue();//要被删除的块
        auto bbjmp=constCond?brInst->iftrue():brInst->iffalse();//要保留的块
        auto bbcur=brInst->parent();//当前的块
        assert(getExecutableFlag(bbcur,bbjmp));
        bbcur->delete_inst(brInst);
        bbcur->emplace_inst(bbcur->insts().end(),new ir::BranchInst(bbjmp,bbcur));
        if(executableFlag[bbcur][bbdel])livePreNum[bbdel]--;
        executableFlag[bbcur][bbdel]=false;//这条边不再可执行  
        if(livePreNum[bbdel]==0){//如果产生了新的死块
            deadFlag[bbdel]=true;
            deleteDeadBlock(bbdel);
        }
        else{//没有产生新的死块
            for(auto inst:bbdel->insts()){
                auto phiInst=dyn_cast<ir::PhiInst>(inst);
                if(not phiInst)break;//要保证phi在块的开头
                phiInst->delbb(bbcur);
                if(phiInst->getConstantRepl()){
                    worklist.push_back(inst);
                }
            }
        }


    }

    void SCCP::deleteDeadBlock(ir::BasicBlock* bb){//删除死块
        assert(deadFlag[bb]);
        // bb->parent()->print(std::cout);
        for(auto puseIter=bb->uses().begin();puseIter!=bb->uses().end();){//将所有的使用bb的phi进行删除其中的bb以及对应的val
            auto puse=*puseIter;
            puseIter++;
            auto puser=puse->user();
            // auto brInst=dyn_cast<ir::BranchInst>(puser);
            // if(brInst){
            //     assert(getDeadFlag(brInst->parent()));
            // }
            auto phiInst=dyn_cast<ir::PhiInst>(puser);
            if(phiInst){
                //将phi中对应bb的参数删除,加入到worklist中!
                phiInst->delbb(bb);
                if(phiInst->getConstantRepl())
                    worklist.push_back(dyn_cast<ir::Instruction>(phiInst));
            }
            // bb->parent()->print(std::cout);
        }
        for(auto bbnext:bb->next_blocks()){//将自己的后继边删除,看看有无产生新的块
            if(executableFlag[bb][bbnext])livePreNum[bbnext]--;
            executableFlag[bb][bbnext]=false;
            if(not livePreNum[bbnext]){//产生的是死块
                deadFlag[bbnext]=true;
                deleteDeadBlock(bbnext);
            }
        }

    }
}