#pragma once

#include "mir/mir.hpp"
// #include "mir/lowering.hpp"
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
class LoweringContext;
class TargetFrameInfo {
   public:
    TargetFrameInfo() = default;
    virtual ~TargetFrameInfo() = default;
    // lowering stage
    virtual void emit_call(ir::CallInst* inst, LoweringContext& lowering_ctx) = 0;
    virtual void emit_prologue(MIRFunction* func,
                               LoweringContext& lowering_ctx) = 0;
    virtual void emit_return(ir::ReturnInst* inst, LoweringContext& lowering_ctx) = 0;
    // ra stage
    virtual bool is_caller_saved(MIROperand& op) = 0;
    virtual bool is_callee_saved(MIROperand& op) = 0;
    // sa stage
    virtual int stack_pointer_align() = 0;
    virtual void emit_postsa_prologue(MIRBlock* entry, int32_t stack_size) = 0;
    virtual void emit_postsa_epilogue(MIRBlock* exit, int32_t stack_size) = 0;
    // virtual int32_t insert_prologue_epilogue(MIRFunction* func);
};

}  // namespace mir
