#include "pass/optimize/TCO.hpp"
using namespace pass;

void tailCallOpt::run(ir::Function* func,topAnalysisInfoManager* tp){
    if(func->isOnlyDeclare())return;
    bool isChange=false;
    std::vector<ir::CallInst*>tail_call_vec;
    for(auto bb:func->blocks()){
        for(auto inst:bb->insts()){
            if(auto callInstPtr=dyn_cast<ir::CallInst>(inst)){
                if(callInstPtr->callee()==func and is_tail_rec(inst,func))
                    tail_call_vec.push_back(callInstPtr);
                else if(is_tail_call(inst,func));// for backend
                    callInstPtr->setIsTail(true);
            }
        }
    }
    if(tail_call_vec.empty())
        return;
    std::cerr<<"Function \""<<func->name()<<"\" is tail-recursive, have "<<tail_call_vec.size()<<" tail calls."<<std::endl;
    isChange=true;
    //首先要在func的entry前再添加一个entry,以及对应的无条件跳转指令
    auto newEntry=new ir::BasicBlock("newbb",func);
    func->blocks().push_front(newEntry);
    auto oldEntry=func->entry();
    ir::BasicBlock::block_link(newEntry,oldEntry);
    auto newBrInst=new ir::BranchInst(oldEntry,newEntry);
    newEntry->emplace_first_inst(newBrInst);
    func->setEntry(newEntry);
    //在oldEntry添加关于参数的phi
    for(auto argIter=func->args().rbegin();argIter!=func->args().rend();argIter++){
        auto arg=*argIter;
        auto newPhi=new ir::PhiInst(oldEntry,arg->type());
        oldEntry->emplace_first_inst(newPhi);
        newPhi->addIncoming(arg,newEntry);
        for(auto puseIter=arg->uses().begin();puseIter!=arg->uses().end();){
            auto puse=*puseIter;
            puseIter++;
            auto puserPhiInst=dyn_cast<ir::PhiInst>(puse->user());
            if(puserPhiInst==newPhi)continue;
            auto user=puse->user();
            user->setOperand(puse->index(),newPhi);
        }
    }
    //对于每一个尾调用call指令生成相应的无条件跳转指令
    for(auto tail_callInst:tail_call_vec){
        for(auto bbnextIter=tail_callInst->block()->next_blocks().begin();bbnextIter!=tail_callInst->block()->next_blocks().end();){
            auto bbnext=*bbnextIter;
            bbnextIter++;
            ir::BasicBlock::delete_block_link(tail_callInst->block(),bbnext);
        }
        auto& curBBInsts=tail_callInst->block()->insts();
        auto callInstPos=std::find(curBBInsts.begin(),curBBInsts.end(),tail_callInst);
        auto newBrInst=new ir::BranchInst(oldEntry,tail_callInst->block());
        tail_callInst->block()->emplace_inst(callInstPos,newBrInst);
        ir::BasicBlock::block_link(tail_callInst->block(),oldEntry);
        auto rargIter=tail_callInst->rargs().begin();
        for(auto pinst:oldEntry->phi_insts()){
            auto phiinst=dyn_cast<ir::PhiInst>(pinst);
            phiinst->addIncoming((*rargIter)->value(),tail_callInst->block());
            rargIter++;
        }
        recursiveDeleteInst(tail_callInst);
    }
    if(isChange){
        tp->CFGChange(func);
        tp->CallChange();
    }
    
}

bool tailCallOpt::is_tail_rec(ir::Instruction* inst,ir::Function* func){
    auto& bbInsts=inst->block()->insts();
    auto instPos=std::find(bbInsts.begin(),bbInsts.end(),inst);
    auto instValueId=inst->valueId();
    if(instValueId==ir::vCALL){
        auto clinst=dyn_cast<ir::CallInst>(inst);
        instPos++;
        return func==clinst->callee() and is_tail_rec(*instPos,func);
    }
    else if(instValueId==ir::vBR){
        auto brinst=dyn_cast<ir::BranchInst>(inst);
        if(brinst->is_cond()){
            auto trueBlockInst=brinst->iftrue()->insts().front();
            auto falseBlockInst=brinst->iffalse()->insts().front();
            return is_tail_rec(trueBlockInst,func) and is_tail_rec(falseBlockInst,func);
        }
        else{
            auto destBlockInst=brinst->dest()->insts().front();
            return is_tail_rec(destBlockInst,func);
        }
    }
    else if(instValueId==ir::vRETURN)
        return true;
    else if(ir::vPHI==instValueId){
        instPos++;
        return is_tail_rec(*instPos,func);
    }
    else
        return false;    
}

bool tailCallOpt::is_tail_call(ir::Instruction* inst,ir::Function* func){
    auto& bbInsts=inst->block()->insts();
    auto instPos=std::find(bbInsts.begin(),bbInsts.end(),inst);
    auto instValueId=inst->valueId();
    if(instValueId==ir::vCALL){
        auto clinst=dyn_cast<ir::CallInst>(inst);
        instPos++;
        return is_tail_call(*instPos,func);
    }
    else if(instValueId==ir::vBR){
        auto brinst=dyn_cast<ir::BranchInst>(inst);
        if(brinst->is_cond()){
            auto trueBlockInst=brinst->iftrue()->insts().front();
            auto falseBlockInst=brinst->iffalse()->insts().front();
            return is_tail_call(trueBlockInst,func) and is_tail_call(falseBlockInst,func);
        }
        else{
            auto destBlockInst=brinst->dest()->insts().front();
            return is_tail_call(destBlockInst,func);
        }
    }
    else if(instValueId==ir::vRETURN)
        return true;
    else if(ir::vPHI==instValueId){
        instPos++;
        return is_tail_call(*instPos,func);
    }
    else
        return false;
}

void tailCallOpt::recursiveDeleteInst(ir::Instruction* inst){
    auto instBB=inst->block();
    auto instIter=std::find(instBB->insts().begin(),instBB->insts().end(),inst);
    ir::BranchInst* brInst;
    for(;instIter!=instBB->insts().end();){
        auto curInst=*instIter;
        instIter++;
        brInst=dyn_cast<ir::BranchInst>(curInst);
        if(brInst!=nullptr)
            break;
        instBB->force_delete_inst(curInst);
    }
    if(brInst->is_cond()){
        auto trueTarget=brInst->iftrue();
        auto falseTarget=brInst->iffalse();
        for(auto pinstIter=trueTarget->phi_insts().begin();pinstIter!=trueTarget->phi_insts().end();){
            auto pinst=*pinstIter;
            pinstIter++;
            auto phiinst=dyn_cast<ir::PhiInst>(pinst);
            phiinst->delBlock(instBB);
            if(phiinst->getsize()==1){
                phiinst->replaceAllUseWith(phiinst->getValue(0));
                trueTarget->force_delete_inst(phiinst);
            }
                
        }
        for(auto pinstIter=falseTarget->phi_insts().begin();pinstIter!=falseTarget->phi_insts().end();){
            auto pinst=*pinstIter;
            pinstIter++;
            auto phiinst=dyn_cast<ir::PhiInst>(pinst);
            phiinst->delBlock(instBB);
            if(phiinst->getsize()==1){
                phiinst->replaceAllUseWith(phiinst->getValue(0));
                falseTarget->force_delete_inst(phiinst);
            }
        }
        recursiveDeleteBB(trueTarget);
        recursiveDeleteBB(falseTarget);
        instBB->force_delete_inst(brInst);
    }
    else{
        auto destTarget=brInst->dest();
        for(auto pinstIter=destTarget->phi_insts().begin();pinstIter!=destTarget->phi_insts().end();){
            auto pinst=*pinstIter;
            pinstIter++;
            auto phiinst=dyn_cast<ir::PhiInst>(pinst);
            phiinst->delBlock(instBB);
            if(phiinst->getsize()==1){
                phiinst->replaceAllUseWith(phiinst->getValue(0));
                destTarget->force_delete_inst(phiinst);
            }
        }
        recursiveDeleteBB(destTarget);
        instBB->force_delete_inst(brInst);
    }

}

void tailCallOpt::recursiveDeleteBB(ir::BasicBlock* bb){
    if(bb->pre_blocks().size()!=0)return;
    std::vector<ir::BasicBlock*>worklist;
    worklist.push_back(bb);
    while(not worklist.empty()){
        auto bbdel=worklist.back();
        worklist.pop_back();
        for(auto bbnext:bb->next_blocks()){
            if(bbnext->pre_blocks().size()==1)
                worklist.push_back(bbnext);
        }
        bbdel->function()->forceDelBlock(bbdel);
    }
}