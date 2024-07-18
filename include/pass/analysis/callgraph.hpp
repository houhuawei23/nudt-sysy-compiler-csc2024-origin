#pragma once
#include "ir/ir.hpp"
#include "pass/pass.hpp"
#include <vector>
#include <set>

namespace pass{
    class callGraphBuild : public ModulePass{
        public:
            void run(ir::Module* ctx,topAnalysisInfoManager* tp)override;
            
        private:
            std::vector<ir::Function*>funcStack;
            std::set<ir::Function*>funcSet;
            std::map<ir::Function*,bool>vis;
            void dfsFuncCallGraph(ir::Function*func);
        private:
            callGraph* cgctx;

    };

    class callGraphCheck : public ModulePass{
        public:
            void run(ir::Module* ctx, topAnalysisInfoManager* tp)override;
        private:
            callGraph* cgctx;
    };
}