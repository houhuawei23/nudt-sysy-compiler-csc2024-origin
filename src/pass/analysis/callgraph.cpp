#include "pass/analysis/callgraph.hpp"

namespace pass{
    void callGraphBuild::run(ir::Module* ctx){
        for(auto func:ctx->funcs()){//initialize call info for functions
            if(not func->entry())
                func->set_is_lib(true);
            else
                func->set_is_lib(false);
            func->set_is_called(false);
            func->set_is_inline(true);
            func->callees().clear();
            vis[func]=false;
        }

        for(auto func:ctx->funcs()){// travel all inst and collect call information
            for(auto bb:func->blocks()){
                for(auto inst:bb->insts()){
                    auto instCall=dyn_cast<ir::CallInst>(inst);
                    if(instCall){
                        auto calleePtr=instCall->callee();
                        if(calleePtr->get_is_lib())continue;// lib function don't need call info
                        func->callees().insert(calleePtr);
                        func->set_is_called(true);
                    }
                }
            }
        }
        assert(funcStack.empty());
        assert(funcSet.empty());
        dfsFuncCallGraph(ctx->main_func());


    }
    void callGraphBuild::dfsFuncCallGraph(ir::Function*func){
        funcStack.push_back(func);
        funcSet.insert(func);
        for(auto calleeFunc:func->callees()){
            if(funcSet.count(calleeFunc)){//find a back edge
                calleeFunc->set_is_inline(false);
                for(auto funcIter=funcStack.rbegin();*funcIter!=calleeFunc;funcIter++){
                    (*funcIter)->set_is_inline(false);
                }
            }
            else{// normal edge, and we continue recursive
                if(not vis[calleeFunc])
                    dfsFuncCallGraph(calleeFunc);
            }
        }
        funcStack.pop_back();
        funcSet.erase(func);
    }
    std::string callGraphBuild::name(){return "callGraphBuild";}
}