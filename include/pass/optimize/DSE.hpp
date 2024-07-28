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
    class simpleDSE : public BasicBlockPass{
    public:
        void run(ir::BasicBlock* bb,TopAnalysisInfoManager* tp) override;
        
    };
}