#pragma once
#include "mir/mir.hpp"
#include <unordered_set>

/*
    1. lowering stage
        - emit_call
        - emit_prologue
        - emit_return

    2. ra stage - register allocation
        - is_caller_saved(MIROperand& op)
        - is_callee_saved(MIROperand& op)

    3. sa stage - stack allocation
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
public:  // lowering stage
    virtual void emit_call(ir::CallInst* inst, LoweringContext& lowering_ctx) = 0;
    virtual void emit_prologue(MIRFunction* func, LoweringContext& lowering_ctx) = 0;
    virtual void emit_return(ir::ReturnInst* inst, LoweringContext& lowering_ctx) = 0;
public:  // ra stage (register allocation)
    virtual bool is_caller_saved(MIROperand& op) = 0;
    virtual bool is_callee_saved(MIROperand& op) = 0;
public:  // sa stage (stack allocation)
    virtual int stack_pointer_align() = 0;
    virtual void emit_postsa_prologue(MIRBlock* entry, int32_t stack_size) = 0;
    virtual void emit_postsa_epilogue(MIRBlock* exit, int32_t stack_size) = 0;
    /* 插入序言和尾声代码: 寄存器保护与恢复 */
    virtual int32_t insert_prologue_epilogue(MIRFunction* func,
                                             std::unordered_set<MIROperand*>& callee_saved_regs,
                                             CodeGenContext& ctx,
                                             MIROperand* return_addr_reg) = 0;
public:  // alignment
    virtual size_t get_stackpointer_alignment() = 0;
};
}