// clang-format off
#pragma once
#include "mir/mir.hpp"
#include "mir/target.hpp"
#include "mir/datalayout.hpp"

#include "target/InstInfoDecl.hpp"
// clang-format on

namespace mir {

class RISCVDataLayout final : public DataLayout {
   public:
    Endian edian() { return Endian::Little; }
    size_t ptr_size() { return 8; }
    size_t code_align() { return 4; }
    size_t mem_align() { return 8; }
};

class RISCVFrameInfo : public TargetFrameInfo {
   public:
    // lowering stage
    void emit_call(ir::CallInst* inst);
    void emit_prologue(MIRFunction* func);
    void emit_return(ir::ReturnInst* inst);
    // ra stage
    bool is_caller_saved(MIROperand& op);
    bool is_callee_saved(MIROperand& op);
    // sa stage
    int stack_pointer_align();
    void emit_postsa_prologue(MIRBlock* entry, int32_t stack_size);
    void emit_postsa_epilogue(MIRBlock* exit, int32_t stack_size);
};

class RISCVTarget : public Target {
    RISCVDataLayout _datalayout;
    RISCVFrameInfo _frameinfo;
    // RISCVRegisterInfo mRegisterInfo;

    const DataLayout& get_datalayout() const override { return _datalayout; }
    const TargetInstInfo& get_inst_info() const override {
        return RISCV::getRISCVInstInfo();
    }

   public:
   RISCVTarget() = default;
};
}  // namespace mir