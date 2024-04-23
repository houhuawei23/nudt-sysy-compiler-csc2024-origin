#pragma once
#include "mir/mir.hpp"
#include <unordered_set>
// #include "mir/target.hpp"

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
// struct CodeGenContext;
class TargetFrameInfo {
   public:
    TargetFrameInfo() = default;
    virtual ~TargetFrameInfo() = default;

   public:  // lowering stage
    virtual void emit_call(ir::CallInst* inst,
                           LoweringContext& lowering_ctx) = 0;
    virtual void emit_prologue(MIRFunction* func,
                               LoweringContext& lowering_ctx) = 0;
    virtual void emit_return(ir::ReturnInst* inst,
                             LoweringContext& lowering_ctx) = 0;

   public:  // ra stage
    virtual bool is_caller_saved(MIROperand& op) = 0;
    virtual bool is_callee_saved(MIROperand& op) = 0;

   public:  // sa stage
    virtual int stack_pointer_align() = 0;
    virtual void emit_postsa_prologue(MIRBlock* entry, int32_t stack_size) = 0;
    virtual void emit_postsa_epilogue(MIRBlock* exit, int32_t stack_size) = 0;
    virtual int32_t insert_prologue_epilogue(
        MIRFunction* func,
        std::unordered_set<MIROperand*>& call_saved_regs,
        CodeGenContext& ctx,
        MIROperand* return_addr_reg);
};
}  // namespace mir
