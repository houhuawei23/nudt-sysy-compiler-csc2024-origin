#pragma once
#include "mir/mir.hpp"
#include "mir/target.hpp"
#include "mir/datalayout.hpp"
// #include "mir/iselinfo.hpp"
#include "mir/registerinfo.hpp"

#include "mir/lowering.hpp"

#include "target/generic/generic.hpp"
#include "target/generic/InstInfoDecl.hpp"

#include "target/generic/ISelInfoDecl.hpp"
// #include "target/generic/ISelInfoImpl.hpp"
namespace mir {


class GNERICDataLayout final : public DataLayout {
    // just for fun
   public:
    Endian edian() override { return Endian::Little; }
    size_t type_align(ir::Type* type) override {
        return 4;  // TODO: check type size
    }
    size_t ptr_size() override { return 8; }
    size_t code_align() override { return 4; }
    size_t mem_align() override { return 8; }
};

class GENERICFrameInfo final : public TargetFrameInfo {
   public:
    // lowering stage
    void emit_call(ir::CallInst* inst, LoweringContext& lowering_ctx) override {
        //
        std::cerr << "RISCV emit_call not implemented" << std::endl;
    }

    // 在函数调用前生成序言代码，用于设置栈帧和保存寄存器状态。
    void emit_prologue(MIRFunction* func,
                       LoweringContext& lowering_ctx) override {
        std::cerr << "GENERIC emit_prologue not implemented" << std::endl;
    }

    void emit_return(ir::ReturnInst* ir_inst,
                     LoweringContext& lowering_ctx) override {
        auto inst = new MIRInst(InstReturn);
        lowering_ctx.emit_inst(inst);
        std::cerr << "GENERIC emit_return not implemented" << std::endl;
    }

    // ra stage
    bool is_caller_saved(MIROperand& op) override {
        std::cerr << "GENERIC is_caller_saved not implemented" << std::endl;
        return true;
    }
    bool is_callee_saved(MIROperand& op) override {
        std::cerr << "GENERIC is_callee_saved not implemented" << std::endl;
        return true;
    }
    // sa stage
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

class GENERICTarget final : public Target {
    GNERICDataLayout _datalayout;
    GENERICFrameInfo _frameinfo;

   public:
    GENERICTarget() = default;
    //! get
    DataLayout& get_datalayout() override { return _datalayout; }

    TargetInstInfo& get_target_inst_info() override {
        return GENERIC::getGENERICInstInfo();
    }

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

    // emit_assembly
    void emit_assembly(std::ostream& out, MIRModule& module) override;
};

}  // namespace mir
