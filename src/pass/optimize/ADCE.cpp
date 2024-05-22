#include "pass/optimize/ADCE.hpp"

static std::set<ir::Instruction*>workList;

namespace pass{
    void ADCE::run(ir::Function* func){
        
    }

    bool ADCE::isAlive(ir::Instruction* inst){

    }

    void ADCE::addAlive(ir::Instruction*inst){

    }
}