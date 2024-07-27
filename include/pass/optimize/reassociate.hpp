#pragma once
#include "ir/ir.hpp"
#include "pass/pass.hpp"
namespace pass{
    class reassociatePass : public FunctionPass{
        public:
            void run(ir::Function* func, TopAnalysisInfoManager* tp) override;
        private:
            void DFSPostOrderBB(ir::BasicBlock* bb);
            void buildRankMap();
    }; 
}
