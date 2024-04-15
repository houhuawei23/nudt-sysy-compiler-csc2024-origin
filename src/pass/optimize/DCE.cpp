#include "pass/optimize/DCE.hpp"

static std::set<ir::Instruction*>alive;

namespace pass{
    
    void DCE::run(ir::Function* func){
        if(!func->entry())return;
        for(auto bb:func->blocks()){//扫描所有的指令,只要是isAlive的就加到alive列表中
            for(auto inst:bb->insts()){
                if(isAlive(inst))addAlive(inst);
            }
        }

        for(auto bb:func->blocks()){
            for(auto instIter=bb->insts().begin();instIter!=bb->insts().end();){
                auto curIter=*instIter;
                instIter++;
                if(alive.count(curIter)==0)
                    bb->force_delete_inst(curIter);
            }
        }
    }

    bool DCE::isAlive(ir::Instruction* inst){//只有store,terminator和call inst是活的
        return inst->is_noname() or dyn_cast<ir::CallInst>(inst);
    }

    void DCE::addAlive(ir::Instruction*inst){//递归的将活代码和他的依赖加入到alive列表当中
        alive.insert(inst);
        for(auto op:inst->operands()){
            auto opInst=dyn_cast<ir::Instruction>(op->value());
            if(opInst and !alive.count(opInst))
                addAlive(opInst);
        }
    }

}