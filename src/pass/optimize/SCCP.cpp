#include "pass/optimize/SCCP.hpp"
/*
SCCP的流程如下

*/

static std::vector<ir::Instruction*>inst_worklist;
static std::set<ir::BasicBlock*>cfg_worklist;

namespace pass{
    std::string SCCP::name(){return "SCCP";}

    void SCCP::run(ir::Function* func){
        if(!func->entry())return;
        
    }
    
}