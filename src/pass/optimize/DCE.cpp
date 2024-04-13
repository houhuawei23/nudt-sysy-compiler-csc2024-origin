#include "pass/optimize/DCE.hpp"

static std::set<ir::Instruction*>alive;

namespace pass{
    
    void DCE::run(ir::Function* func){
        if(!func->entry())return;
        for(auto bb:func->blocks()){
            for(auto inst:bb->insts()){
                if(isAlive(inst))addAlive(inst);
            }
        }
        for(auto bb:func->blocks()){
            for(auto instIter=bb->insts().begin();instIter!=bb->insts().end();){
                auto curIter=instIter++;
                if(alive.count(*curIter)==0)bb->delete_inst(*curIter);
            }
        }

    }

    bool DCE::isAlive(ir::Instruction* inst){//只有store,terminator和call inst是活的
        return inst->is_noname() or dyn_cast<ir::CallInst>(inst);
    }

    void DCE::addAlive(ir::Instruction*inst){
        alive.insert(inst);
        for(auto op:inst->operands()){
            auto opInst=dyn_cast<ir::Instruction>(op->value());
            if(opInst and alive.count(opInst))
                addAlive(opInst);
        }
    }
}