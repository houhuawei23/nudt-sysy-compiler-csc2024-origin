#pragma once
#include <set>
#include <cassert>
#include <map>
#include <vector>
#include <algorithm>
#include <queue>
#include "ir/ir.hpp"
#include "pass/pass.hpp"

namespace pass{
    class tailCallOpt : public FunctionPass{
        public:
            void run(ir::Function* func, topAnalysisInfoManager* tp)override;
        private:
            bool is_tail_call(ir::Instruction* inst,ir::Function* func);
    };
}