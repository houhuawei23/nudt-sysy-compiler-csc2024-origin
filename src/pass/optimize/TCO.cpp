#include "pass/optimize/TCO.hpp"
using namespace pass;

void tailCallOpt::run(ir::Function* func,topAnalysisInfoManager* tp){
    std::vector<ir::CallInst*>tail_call_vec;
    for(auto bb:func->blocks()){
        for(auto inst:bb->insts()){
            if(auto callInstPtr=dyn_cast<ir::CallInst>(inst)){
                if(callInstPtr->callee()==func and is_tail_call(inst,func))
                    tail_call_vec.push_back(callInstPtr);
            }
        }
    }
    if(tail_call_vec.empty())
        return;
    std::cerr<<"Function \""<<func->name()<<"\" is tail-recursive, have "<<tail_call_vec.size()<<" tail calls."<<std::endl;
    
}

bool tailCallOpt::is_tail_call(ir::Instruction* inst,ir::Function* func){
    auto& bbInsts=inst->block()->insts();
    auto instPos=std::find(bbInsts.begin(),bbInsts.end(),inst);
    auto instValueId=inst->valueId();
    if(instValueId==ir::vCALL){
        auto clinst=dyn_cast<ir::CallInst>(inst);
        instPos++;
        return func==clinst->callee() and func->args().size()<30 
        and is_tail_call(*instPos,func);
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