#pragma once
#include <set>
#include <cassert>
#include <map>
#include <vector>
#include <queue>
#include "ir/ir.hpp"
#include "pass/pass.hpp"

namespace pass{
    class SCCP:public FunctionPass{
        public:
            void run(ir::Function* func, topAnalysisInfoManager* tp)override;
            
        private:
            using FlowEdge=std::pair<ir::BasicBlock*,ir::BasicBlock*>;
            using SSAEdge=std::pair<ir::Instruction*,ir::Instruction*>;
            static std::queue<FlowEdge>FlowWorkList;
            static std::queue<SSAEdge>SSAWorkList;
            using valueEdge=std::pair<ir::Value*,ir::Value*>;
            valueEdge getEdge();
    };
}