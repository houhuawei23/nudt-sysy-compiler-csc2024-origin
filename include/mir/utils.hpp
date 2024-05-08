#pragma once
#include "mir/mir.hpp"
#include "mir/target.hpp"

#include <iostream>
#include <functional>

namespace mir {
/* dump assembly code for a module */
void dump_assembly(std::ostream& os, MIRModule& module, CodeGenContext& ctx);

/* allocate stack space for local variables */
void allocateStackObjects(MIRFunction* func, CodeGenContext& ctx);

/* for each def operand in a block, apply functor */
void forEachDefOperand(MIRBlock& block,
                       CodeGenContext& ctx,
                       const std::function<void(MIROperand* op)>& functor);

/* for each def operand in a function, apply functor */
void forEachDefOperand(MIRFunction& func,
                       CodeGenContext& ctx,
                       const std::function<void(MIROperand* op)>& functor);

bool genericPeepholeOpt(MIRFunction& func, CodeGenContext& ctx);

void postLegalizeFunc(MIRFunction& func, CodeGenContext& ctx);

/** Schedule */
void preRASchedule(MIRFunction& func, const CodeGenContext& ctx);
void postRASchedule(MIRFunction& func, const CodeGenContext& ctx);
}  // namespace mir
