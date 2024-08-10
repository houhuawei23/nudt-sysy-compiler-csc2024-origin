#pragma once
#include "ir/ir.hpp"
#include "pass/pass.hpp"
#include "pass/analysis/dependenceAnalysis/DependenceAnalysis.hpp"

namespace pass{
    ir::Value* addrToBaseaddr(ir::Value* ptr);
    baseAddrType getBaseaddrType(ir::Value* ptr);
    bool isBaseAddrPossiblySame(ir::Value* ptr1,ir::Value* ptr2,ir::Function* func,callGraph* cgctx);
};

