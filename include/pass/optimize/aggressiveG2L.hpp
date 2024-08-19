#pragma once
#include <set>
#include <cassert>
#include <map>
#include <vector>
#include "ir/ir.hpp"
#include "pass/pass.hpp"
namespace pass{
    class aggressiveG2L:public ModulePass{
        public:
            void run(ir::Module* md,TopAnalysisInfoManager* tp)override;
            std::string name()const override{return "AG2L";}
        private:
            domTree* domctx;
            sideEffectInfo* sectx;
            callGraph* cgctx;
            void replaceReadOnlyGv(ir::GlobalVariable* gv);
            void deleteWriteOnlyGv(ir::GlobalVariable* gv);
            void replaceGvInMain(ir::GlobalVariable* gv,ir::Function* func);//配合mem2reg使用
            void replaceGvInNormalFunc(ir::GlobalVariable* gv,ir::Function* func);//配合mem2reg使用
            void replaceGvInOneFunc(ir::GlobalVariable* gv,ir::Function* func);//配合mem2reg使用

    };
}