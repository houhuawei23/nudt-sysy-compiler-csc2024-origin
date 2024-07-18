#pragma once
#include <vector>
#include <unordered_map>
#include "mir/mir.hpp"
#include "mir/target.hpp"

/*
 * @brief: CFG Analysis
 * @note: 分析MIR控制流图
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
    std::unordered_map<MIRBlock*, MIRBlockCFGInfo> _block2CFGInfo;
public:  // get function
    std::unordered_map<MIRBlock*, MIRBlockCFGInfo>& block2CFGInfo() { return _block2CFGInfo; }
    const std::vector<MIRBlockEdge>& predecessors(MIRBlock* block) const {
        return _block2CFGInfo.at(block).predecessors;
    }
    const std::vector<MIRBlockEdge>& successors(MIRBlock* block) const {
        return _block2CFGInfo.at(block).successors;
    }
public:  // Just for Debug
    void dump(std::ostream& out);
};
CFGAnalysis calcCFG(MIRFunction& mfunc, CodeGenContext& ctx);

struct BlockTripCountResult final {
    std::unordered_map<MIRBlock*, double> _freq;
public:
    auto& storage() { return _freq; }
    double query(MIRBlock* block) const;
};
BlockTripCountResult calcFreq(MIRFunction& mfunc, CFGAnalysis& cfg);
}