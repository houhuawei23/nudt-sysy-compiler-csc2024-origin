#pragma once
#include "mir/mir.hpp"
#include "mir/target.hpp"
#include "mir/datalayout.hpp"
#include "mir/registerinfo.hpp"
#include "mir/lowering.hpp"

#include "target/generic/generic.hpp"
#include "target/generic/InstInfoDecl.hpp"
#include "target/generic/ISelInfoDecl.hpp"

namespace mir {
/*
 * @brief: GENERICDataLayout Class
 * @note: 
 *      MIR Generic Data Layout
 */
class GENERICDataLayout final : public DataLayout {
    public:
        Endian edian() override { return Endian::Little; }
        size_t ptr_size() override { return 8; }

    public:  // align
        size_t type_align(ir::Type* type) override {
            return 4;  // TODO: check type size
        }
        size_t code_align() override { return 4; }
        size_t mem_align() override { return 8; }
};

/*
 * @brief: GENERICFrameInfo Class
 * @note: 
 *      MIR Generic Frame Information
 */
class GENERICFrameInfo final : public TargetFrameInfo {
    public:  // lowering stage
        void emit_call(ir::CallInst* inst, LoweringContext& lowering_ctx) override {
            std::cerr << "GENERIC emit_call not implemented" << std::endl;
        }
        // 在函数调用前生成序言代码，用于设置栈帧和保存寄存器状态。
        void emit_prologue(MIRFunction* func, LoweringContext& lowering_ctx) override {
            std::cerr << "GENERIC emit_prologue not implemented" << std::endl;
        }
        void emit_return(ir::ReturnInst* ir_inst,
                        LoweringContext& lowering_ctx) override {
            auto inst = new MIRInst(InstReturn);
            lowering_ctx.emit_inst(inst);
            std::cerr << "GENERIC emit_return not implemented" << std::endl;
        }

    public:  // ra stage
        bool is_caller_saved(MIROperand& op) override {
            std::cerr << "GENERIC is_caller_saved not implemented" << std::endl;
            return true;
        }
        bool is_callee_saved(MIROperand& op) override {
            std::cerr << "GENERIC is_callee_saved not implemented" << std::endl;
            return true;
        }
    
    public:  // sa stage
        int stack_pointer_align() override {
            std::cerr << "GENERIC stack_pointer_align not implemented" << std::endl;
            return 8;
        }
        void emit_postsa_prologue(MIRBlock* entry, int32_t stack_size) override {
            std::cerr << "GENERIC emit_postsa_prologue not implemented"
                    << std::endl;
        }
        void emit_postsa_epilogue(MIRBlock* exit, int32_t stack_size) override {
            std::cerr << "GENERIC emit_postsa_epilogue not implemented"
                    << std::endl;
        }
};

/*
 * @brief: GENERICTarget Class
 * @note: 
 *      继承自Target抽象基类, Generic Target (通用架构)
 */
class GENERICTarget final : public Target {
    GENERICDataLayout _datalayout;
    GENERICFrameInfo _frameinfo;

    public:
        GENERICTarget() = default;
    
    public:  // get
        DataLayout& get_datalayout() override { return _datalayout; }
        TargetInstInfo& get_target_inst_info() override { return GENERIC::getGENERICInstInfo(); }
        TargetISelInfo& get_target_isel_info() override {
            std::cerr << "Not Impl get_isel_info" << std::endl;
            // assert(false);
            return GENERIC::getGENERICISelInfo();
        }
        TargetRegisterInfo& get_register_info() override {
            std::cerr << "Not Impl get_register_info" << std::endl;
            assert(false);
        }
        TargetFrameInfo& get_target_frame_info() override { return _frameinfo; }

    public:  // emit_assembly
        void emit_assembly(std::ostream& out, MIRModule& module) override;
};

class GENERICRegisterInfo : public TargetRegisterInfo {
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
    bool is_zero_reg() {
        std::cerr << "Not Impl is_zero_reg" << std::endl;
        return false;
    }
};

}  // namespace mir
