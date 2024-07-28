#pragma once
#include <set>
#include <cassert>
#include <map>
#include <vector>
#include "ir/ir.hpp"
#include "pass/pass.hpp"

namespace pass{
    class global2local:public ModulePass{
        private:
            callGraph* cgctx;
            void globalCallAnalysis(ir::Module* md);
            void addIndirectGlobalUseFunc(ir::GlobalVariable* gv, ir::Function* func);
            bool processGlobalVariables(ir::GlobalVariable* gv,ir::Module* md,TopAnalysisInfoManager* tp);
        public:
            void run(ir::Module* md,TopAnalysisInfoManager* tp)override;
    };
}