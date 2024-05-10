#pragma once
#include "pass/pass.hpp"

namespace pass{
    class loopAnalysis:public FunctionPass{
        public:
            std::string name()override;
            void run(ir::Function* func)override;
        private:
            void addLoopBlocks(ir::Function*func,ir::BasicBlock* header,ir::BasicBlock* tail);
    };

    class loopInfoCheck:public FunctionPass{
        std::string name()override;
        void run(ir::Function* func)override;
    };
}
