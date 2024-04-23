// clang-format off
#pragma once
#include "mir/mir.hpp"
#include "mir/utils.hpp"
#include "mir/target.hpp"
#include "mir/datalayout.hpp"
#include "mir/registerinfo.hpp"

#include "target/riscv/riscv.hpp"
#include "target/riscv/InstInfoDecl.hpp"
#include "target/riscv/ISelInfoDecl.hpp"
#include "target/riscv/ISelInfoImpl.hpp"
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
    void emit_call(ir::CallInst* inst, LoweringContext& lowering_ctx) override {
        // TODO: implement emit call

        std::cerr << "RISCV emit_call not implemented" << std::endl;
    }

    // 在函数调用前生成序言代码，用于设置栈帧和保存寄存器状态。
    void emit_prologue(MIRFunction* func,
                       LoweringContext& lowering_ctx) override {
        // TODO: implement prologue
        std::cerr << "RISCV prologue not implemented" << std::endl;
    }

    void emit_return(ir::ReturnInst* ir_inst,
                     LoweringContext& lowering_ctx) override {
        // TODO: implement emit return
        if (not ir_inst->operands().empty()) {
            // has return value
            // TODO
            auto retval = ir_inst->return_value();
            if (retval->type()->is_float()) {
                lowering_ctx.emit_copy(
                    MIROperand::as_isareg(RISCV::F10, OperandType::Float32),
                    lowering_ctx.map2operand(retval));
            } else if (retval->type()->is_int()) {
                lowering_ctx.emit_copy(
                    MIROperand::as_isareg(RISCV::X10, OperandType::Int64),
                    lowering_ctx.map2operand(retval));
            }
        }
        auto inst = new MIRInst(InstReturn);
        lowering_ctx.emit_inst(inst);
    }

    // ra stage
    bool is_caller_saved(MIROperand& op) override { return true; }
    bool is_callee_saved(MIROperand& op) override { return true; }
    // sa stage
    int stack_pointer_align() override { return 8; }
    void emit_postsa_prologue(MIRBlock* entry, int32_t stack_size) override {
        std::cerr << "Not Impl emit_postsa_prologue" << std::endl;
    }
    void emit_postsa_epilogue(MIRBlock* exit, int32_t stack_size) override {
        std::cerr << "Not Impl emit_postsa_epilogue" << std::endl;
    }
    int32_t insert_prologue_epilogue(
        MIRFunction* func,
        std::unordered_set<MIROperand*>& call_saved_regs,
        CodeGenContext& ctx,
        MIROperand* return_addr_reg) override {
        std::cerr << "Not Impl insert_prologue_epilogue" << std::endl;
        return 0;
    }
};

class RISCVRegisterInfo : public TargetRegisterInfo {
    uint32_t get_alloca_class_cnt() {
        std::cerr << "Not Impl get_alloca_class_cnt" << std::endl;
        return 0;
    }
    uint32_t get_alloca_class(OperandType type) {
        std::cerr << "Not Impl get_alloca_class" << std::endl;
        return 0;
    }

    bool is_legal_isa_reg_operand(MIROperand& op) {
        std::cerr << "Not Impl is_legal_isa_reg_operand" << std::endl;
        return false;
    }
    // ..
    bool is_zero_reg() {
        std::cerr << "Not Impl is_zero_reg" << std::endl;
        return false;
    }
    OperandType getCanonicalizedRegisterType(OperandType type) {
        std::cerr << "Not Impl getCanonicalizedRegisterType" << std::endl;
        return type;
    }
};

class RISCVTarget : public Target {
    RISCVDataLayout _datalayout;
    RISCVFrameInfo _frameinfo;
    RISCVRegisterInfo mRegisterInfo;

   public:
    RISCVTarget() = default;
    //! getXXX
    DataLayout& get_datalayout() override { return _datalayout; }
    TargetInstInfo& get_target_inst_info() override {
        return RISCV::getRISCVInstInfo();
    }
    TargetISelInfo& get_target_isel_info() override {
        std::cerr << "Not Impl get_isel_info" << std::endl;
        return RISCV::getRISCVISelInfo();
    }
    TargetFrameInfo& get_target_frame_info() override { return _frameinfo; }
    TargetRegisterInfo& get_register_info() override { return mRegisterInfo; }

    // emit_assembly
    void emit_assembly(std::ostream& out, MIRModule& module) override {
        std::cerr << "Not Impl emit_assembly" << std::endl;
        auto& target = *this;
        CodeGenContext codegen_ctx{
            target, target.get_datalayout(), target.get_target_inst_info(),
            target.get_target_frame_info(), MIRFlags{false, false}};
        dump_assembly(out, module, codegen_ctx);
    }
};
}  // namespace mir