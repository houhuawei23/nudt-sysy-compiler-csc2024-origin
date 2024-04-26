#pragma once

#include "mir/mir.hpp"
#include "mir/utils.hpp"
#include "mir/target.hpp"
#include "mir/datalayout.hpp"
#include "mir/registerinfo.hpp"
#include "mir/lowering.hpp"

#include "target/riscv/riscv.hpp"
#include "autogen/riscv/InstInfoDecl.hpp"
#include "autogen/riscv/ISelInfoDecl.hpp"
// clang-format on

namespace mir {
/*
 * @brief: RISCVDataLayout Class
 * @note:
 *      RISC-V DataLayout (特定于相关架构) -- 数据信息
 */
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

/*
 * @brief: RISCVFrameInfo Class
 * @note:
 *      RISC-V Frame Information (特定于相关架构) -- 帧信息
 */
class RISCVFrameInfo : public TargetFrameInfo {
   public:  // lowering stage
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
        return;
        // TODO: implement emit return
        if (not ir_inst->operands().empty()) {  // has return value
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

/*
 * @brief: RISCVRegisterInfo Class
 * @note:
 *      RISC-V Register Information (特定于相关架构)
 */
class RISCVRegisterInfo : public TargetRegisterInfo {
    // get function
    uint32_t get_alloca_class_cnt() {
        std::cerr << "Not Impl get_alloca_class_cnt" << std::endl;
        return 0;
    }
    uint32_t get_alloca_class(OperandType type) {
        std::cerr << "Not Impl get_alloca_class" << std::endl;
        return 0;
    }

    MIROperand* get_return_address_register() { return RISCV::ra; }
    MIROperand* get_stack_pointer_register() { return RISCV::sp; }

    // check function
    bool is_legal_isa_reg_operand(MIROperand& op) {
        std::cerr << "Not Impl is_legal_isa_reg_operand" << std::endl;
        return false;
    }
    bool is_zero_reg(const uint32_t x) const {
        return x == RISCV::RISCVRegister::X0;
    }
    OperandType getCanonicalizedRegisterType(OperandType type) {
        std::cerr << "Not Impl getCanonicalizedRegisterType" << std::endl;
        return type;
    }
};

/*
 * @brief: RISCVTarget Class
 * @note:
 *      RISC-V Target (特定于相关架构)
 */
class RISCVTarget : public Target {
    RISCVDataLayout _datalayout;
    RISCVFrameInfo _frameinfo;
    RISCVRegisterInfo _mRegisterInfo;

   public:
    RISCVTarget() = default;

   public:  // get function
    DataLayout& get_datalayout() override { return _datalayout; }
    TargetFrameInfo& get_target_frame_info() override { return _frameinfo; }
    TargetRegisterInfo& get_register_info() override { return _mRegisterInfo; }
    TargetInstInfo& get_target_inst_info() override {
        return RISCV::getRISCVInstInfo();
    }
    TargetISelInfo& get_target_isel_info() override {
        std::cerr << "Not Impl get_isel_info" << std::endl;
        return RISCV::getRISCVISelInfo();
    }

    // emit_assembly
    void emit_assembly(std::ostream& out, MIRModule& module) override;
};
}  // namespace mir