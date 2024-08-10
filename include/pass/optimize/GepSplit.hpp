#pragma once
#include "ir/ir.hpp"
#include "pass/pass.hpp"

namespace pass {
class GepSplit : public FunctionPass {
    // std::unordered_map<ir::Value>
public:
    void run(ir::Function* func, TopAnalysisInfoManager* tp) override;
    std::string name() const override { return "GepSplit"; }
private:
    void split_pointer(ir::GetElementPtrInst* inst, 
                       ir::BasicBlock* insertBlock,
                       ir::inst_iterator insertPos);
    void split_array(ir::inst_iterator begin, 
                     ir::BasicBlock* insertBlock, 
                     ir::inst_iterator end);
};
}