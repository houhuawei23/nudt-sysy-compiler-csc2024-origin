// clang-format off
#pragma once
#include "mir/mir.hpp"
#include "mir/target.hpp"
#include "mir/datalayout.hpp"

#include "target/InstInfoDecl.hpp"
#include "target/InstInfoImpl.hpp"
// clang-format on

namespace mir {

class RISCVDataLayout final : public DataLayout {
   public:
    Endian edian() override { return Endian::Little; }
    size_t type_align(ir::Type* type) override {
        return 4;  // TODO: check type size
    }
    size_t ptr_size() override { return 8; }
    size_t code_align() override { return 4; }
    size_t mem_align() override { return 8; }
};

class RISCVFrameInfo : public TargetFrameInfo {
   public:
    // lowering stage
    void emit_call(ir::CallInst* inst) override {}
    void emit_prologue(MIRFunction* func,
                       LoweringContext& lowering_ctx) override {
        // TODO: implement prologue
        std::cerr << "RISCV prologue not implemented" << std::endl;
    }
    void emit_return(ir::ReturnInst* inst) override {}
    // ra stage
    bool is_caller_saved(MIROperand& op) override { return true; }
    bool is_callee_saved(MIROperand& op) override { return true; }
    // sa stage
    int stack_pointer_align() override { return 8; }
    void emit_postsa_prologue(MIRBlock* entry, int32_t stack_size) override {}
    void emit_postsa_epilogue(MIRBlock* exit, int32_t stack_size) override {}
};

class RISCVTarget : public Target {
    RISCVDataLayout _datalayout;
    RISCVFrameInfo _frameinfo;
    // RISCVRegisterInfo mRegisterInfo;

    DataLayout& get_datalayout() override { return _datalayout; }
    TargetInstInfo& get_inst_info() override {
        return RISCV::getRISCVInstInfo();
    }
    TargetFrameInfo& get_frame_info() override { return _frameinfo; }

   public:
    RISCVTarget() = default;
};
}  // namespace mir