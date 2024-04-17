#include "pass/optimize/optimize.hpp"
#include "pass/optimize/SCP.hpp"
#include <vector>
//当前是简单常量传播遍
static std::set<ir::Instruction*>worklist;

namespace pass{
    void SCP::run(ir::Function* func){
        if(!func->entry())return;
        // func->print(std::cout);
        for(auto bb:func->blocks()){
            for(auto instIter=bb->insts().begin();instIter!=bb->insts().end();){
                auto curInst=*instIter;
                instIter++;
                if(curInst->is_constprop())worklist.insert(curInst);
            }
        }
        while(!worklist.empty()){
            auto curInst=worklist.begin();
            worklist.erase(curInst);
            addConstFlod(*curInst);
        }
    }

    void SCP::addConstFlod(ir::Instruction* inst){
        auto replval=inst->getConstantRepl();
        for(auto puse:inst->uses()){
            auto puser=puse->user();
            puser->set_operand(puse->index(),replval);
            auto puserInst=dyn_cast<ir::Instruction>(puser);
            assert(puserInst);
            if(puserInst->is_constprop()){
                worklist.insert(puserInst);
            }
        }
        inst->uses().clear();
        inst->parent()->delete_inst(inst);
    }

    std::string SCP::name(){return "SCP";}
}