#pragma once
#include "ir/ir.hpp"
#include "pass/pass.hpp"
#include <set>
#include <queue>

namespace pass
{
    class ADCE : public FunctionPass
    {
        public:
            void run(ir::Function* func, topAnalysisInfoManager* tp)override;
        private:
            pdomTree* pdctx;
    };

} // namespace pass

