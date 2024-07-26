#pragma once
#include "ir/ir.hpp"
#include "pass/pass.hpp"

namespace pass {
class CFGPrinter : public FunctionPass {
public:
    void run(ir::Function* func, topAnalysisInfoManager* tp) override;
};
}