#pragma once
#include "pass/pass.hpp"

namespace pass{
    class indVarAnalysis:public FunctionPass{
        public:
            void run(ir::Function* func,topAnalysisInfoManager *tp)override;
        private:
            loopInfo* lpctx;
            indVarInfo* ivctx;
            void addIndVar(ir::Loop* lp, ir::Constant* mbegin, ir::Constant* mstep, ir::Value* mend, ir::BinaryInst* iterinst, ir::Instruction* cmpinst);
            
    };

}