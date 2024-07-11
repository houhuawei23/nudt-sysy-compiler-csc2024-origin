#pragma once
#include "ir/ir.hpp"
#include "pass/pass.hpp"

namespace pass {
class CFGAnalysis : public ModulePass {
public:
    void run(ir::Module* ctx, topAnalysisInfoManager* tp) override;
    void dump(std::ostream& out, ir::Module* ctx);
};
}