#pragma once
#include <vector>
#include <unordered_map>

#include "mir/mir.hpp"
#include "mir/target.hpp"

/*
 * @brief: CFG Analysis
 * @note: 
 *      分析MIR控制流图
 */
namespace mir {
struct MIRBlockEdge final {
    MIRBlock* block;
    double prob;
};

struct MIRBlockCFGInfo final {
    std::vector<MIRBlockEdge> predecessors;
    std::vector<MIRBlockEdge> successors;
};

class CFGAnalysis final {
    std::unordered_map<MIRBlock*, MIRBlockCFGInfo> _mInfo;

    public:  // get function
        std::unordered_map<MIRBlock*, MIRBlockCFGInfo>& mInfo() { return _mInfo; }

        const std::vector<MIRBlockEdge>& predecessors(MIRBlock* block) const { return _mInfo.at(block).predecessors; }
        const std::vector<MIRBlockEdge>& successors(MIRBlock* block) const { return _mInfo.at(block).successors; }
};

CFGAnalysis calcCFG(MIRFunction& mfunc, CodeGenContext& ctx);
}