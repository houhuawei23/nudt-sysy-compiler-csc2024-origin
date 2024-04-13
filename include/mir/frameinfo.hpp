#pragma once

#include "mir/mir.hpp"

/* cmmc
lowering stage
- emit_call
- emit_prologue
- emit_return

ra stage - register allocation
- is_caller_saved(MIROperand& op)
- is_callee_saved(MIROperand& op)

sa stage - stack allocation
- stack_pointer_align
- emit_postsa_prologue
- emit_postsa_epilogue
- insert_prologue_epilogue

*/
namespace mir {
class TargetFrameInfo {
   public:
    TargetFrameInfo() = default;
    virtual ~TargetFrameInfo() = default;
    // lowering stage
    virtual void emit_call(ir::CallInst* inst);
    virtual void emit_prologue(MIRFunction* func);
    virtual void emit_return(ir::ReturnInst* inst);
    // ra stage
    virtual bool is_caller_saved(MIROperand& op);
    virtual bool is_callee_saved(MIROperand& op);
    // sa stage
    virtual int stack_pointer_align();
    virtual void emit_postsa_prologue(MIRBlock* entry, int32_t stack_size);
    virtual void emit_postsa_epilogue(MIRBlock* exit, int32_t stack_size);
    // virtual int32_t insert_prologue_epilogue(MIRFunction* func);
};

}  // namespace mir
