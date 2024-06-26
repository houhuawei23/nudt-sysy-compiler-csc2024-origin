#include "pass/optimize/optimize.hpp"
#include "pass/optimize/reg2mem.hpp"
#include <set>
#include <cassert>
#include <map>
#include <vector>
#include <queue>
#include <algorithm>
namespace pass {
void Reg2Mem::getallphi(ir::Function* func) {
    for (ir::BasicBlock* bb : func->blocks()) {
        if (bb->phi_insts().empty()) {
            continue;
        }
        else{
            for (auto inst : bb->phi_insts()) {
                ir::PhiInst* phiinst = dyn_cast<ir::PhiInst>(inst);
                allphi.push_back(phiinst);
            }
        }
    }
}
void Reg2Mem::run(ir::Function* func) {
    clear();
    getallphi(func);
    DisjSet();
    for (ir::PhiInst* phiinst : allphi){
        int idx = getindex(phiinst);
        ir::PhiInst* root = allphi[find(idx)];
        if (phiweb.find(root) == phiweb.end()){
            ir::AllocaInst* var = new ir::AllocaInst(phiinst->type());
            allocasToinsert.push_back(var);
            phiweb[root] = var;
        }
    }
    for (auto alloca : allocasToinsert){
        ir::BasicBlock* entry = func->entry();
        entry->emplace_first_inst(alloca);
        alloca->set_parent(entry);
    }
    for (ir::PhiInst* phiinst : allphi){
        ir::BasicBlock* nextbb = phiinst->parent();
        int idx = getindex(phiinst);
        ir::PhiInst* root = allphi[find(idx)];
        ir::AllocaInst* variable = phiweb[root];
        ir::LoadInst* phiload = new ir::LoadInst(variable,phiinst->type(),nullptr);//用这条load替换所有对phiinst的使用
        philoadmap[phiinst] = phiload;//记录phiinst与load的映射
        
        for (size_t i = 0; i < phiinst->getsize(); i++){

            ir::BasicBlock* prebb = phiinst->getbb(i);
            ir::Value* phival = phiinst->getval(i);
            ir::StoreInst* phistore = new ir::StoreInst(phival,variable);
            if (auto inst = dyn_cast<ir::PhiInst>(phival) || phival->type()->is_undef()){//如果是phi或则undef则不插入store
                    continue;
            }
            if(prebb->next_blocks().size() == 1){//如果前驱块只有一个后继，直接在前驱块末尾插入store
                ir::inst_iterator it = --(prebb->insts().end());
                prebb->emplace_inst(it,phistore);
                phistore->set_parent(prebb);
            }
            else{//有多个后继则需要在前驱块与当前块中插入一个新的块，在新块中插入store与br指令
                ir::BasicBlock* newbb = func->new_block();
                newbb->set_parent(func);
                ir::BranchInst* br = new ir::BranchInst(nextbb);
                newbb->emplace_back_inst(phistore);
                newbb->emplace_back_inst(br);
                phistore->set_parent(newbb);
                br->set_parent(newbb);
                ir::Instruction* lastinst = *(--(prebb->insts().end()));
                ir::BranchInst* oldbr = dyn_cast<ir::BranchInst>(lastinst);
                oldbr->replacedest(nextbb,newbb);
                ir::BasicBlock::delete_block_link(prebb,nextbb);
                ir::BasicBlock::block_link(prebb,newbb);
                ir::BasicBlock::block_link(newbb,nextbb);
            }
        }
    }
    //删除phi，并将对phi的使用替换为load
    for (ir::PhiInst* phitoremove : allphi){
        ir::BasicBlock* phibb = phitoremove->parent();
        ir::LoadInst* loadinst = philoadmap[phitoremove];
        phitoremove->replace_all_use_with(loadinst);
        phibb->delete_inst(phitoremove);
        phibb->emplace_first_inst(loadinst);
        loadinst->set_parent(phibb);
    }
}
}  // namespace pass