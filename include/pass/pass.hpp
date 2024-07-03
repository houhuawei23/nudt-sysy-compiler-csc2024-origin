#pragma once
#include <vector>
#include "ir/ir.hpp"

namespace pass {

//! Pass Template
template <typename PassUnit>
class Pass {
   public:
    // pure virtual function, define the api
    virtual void run(PassUnit* pass_unit) = 0;
    virtual std::string name() = 0;
};

// Instantiate Pass Class for Module, Function and BB
using ModulePass = Pass<ir::Module>;
using FunctionPass = Pass<ir::Function>;
using BasicBlockPass = Pass<ir::BasicBlock>;


class PassManager{
    ir::Module* _irModule;
    public:
        PassManager(ir::Module* pm){
            _irModule=pm;
        }
        void run(ModulePass* mp){
            mp->run(_irModule);
        }
        void run(FunctionPass* fp){
            for(auto func:_irModule->funcs()){
                fp->run(func);
            }
        }
         void run(BasicBlockPass* bp){
            for(auto func:_irModule->funcs()){
                for(auto bb:func->blocks()){
                    bp->run(bb);
                }
            }
        }
};


}  // namespace pass