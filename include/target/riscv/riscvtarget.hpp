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
#include "autogen/riscv/ScheduleModelDecl.hpp"
// clang-format on

namespace mir {
/*
 * @brief: RISCVDataLayout Class
 * @note: RISC-V数据信息
 */
class RISCVDataLayout final : public DataLayout {
public:
    Endian edian() override { return Endian::Little; }
    size_t type_align(ir::Type* type) override { return 4; }
    size_t ptr_size() override { return 8; }
    size_t code_align() override { return 4; }
    size_t mem_align() override { return 8; }
};
/*
 * @brief: RISCVFrameInfo Class
 * @note: RISC-V帧相关信息
 */
class RISCVFrameInfo : public TargetFrameInfo {
public:  // lowering stage
    void emit_call(ir::CallInst* inst, LoweringContext& lowering_ctx) override;
    // 在函数调用前生成序言代码，用于设置栈帧和保存寄存器状态。
    void emit_prologue(MIRFunction* func, LoweringContext& lowering_ctx) override;
    void emit_return(ir::ReturnInst* ir_inst, LoweringContext& lowering_ctx) override;
public:  // ra stage (register allocation stage)
    // 调用者保存寄存器
    bool is_caller_saved(MIROperand& op) override {
        const auto reg = op.reg();
        // $ra $t0-$t6 $a0-$a7 $ft0-$ft11 $fa0-$fa7
        return reg == RISCV::X1                         ||
               (RISCV::X5 <= reg && reg <= RISCV::X7)   ||
               (RISCV::X10 <= reg && reg <= RISCV::X17) ||
               (RISCV::X28 <= reg && reg <= RISCV::X31) ||
               (RISCV::F0 <= reg && reg <= RISCV::F7)   ||
               (RISCV::F10 <= reg && reg <= RISCV::F17) ||
               (RISCV::F28 <= reg && reg <= RISCV::F31);
    }
    // 被调用者保存寄存器
    bool is_callee_saved(MIROperand& op) override {
        const auto reg = op.reg();
        // $sp $s0-$s7 $f20-$f30 $gp
        return reg == RISCV::X2                         ||
               (RISCV::X8 <= reg && reg <= RISCV::X9)   ||
               (RISCV::X18 <= reg && reg <= RISCV::X27) ||
               (RISCV::F8 <= reg && reg <= RISCV::F9)   ||
               (RISCV::F18 <= reg && reg <= RISCV::F27) ||
               reg == RISCV::X3;
    }
public:  // sa stage (stack allocation stage)
    int stack_pointer_align() override { return 8; }
    void emit_postsa_prologue(MIRBlock* entry, int32_t stack_size) override {
        auto& insts = entry->insts();
        RISCV::adjust_reg(insts, insts.begin(), RISCV::sp, RISCV::sp, -stack_size);
    }
    void emit_postsa_epilogue(MIRBlock* exit, int32_t stack_size) override {
        auto& insts = exit->insts();
        RISCV::adjust_reg(insts, std::prev(insts.end()), RISCV::sp, RISCV::sp, stack_size);
    }
    int32_t insert_prologue_epilogue(MIRFunction* func, std::unordered_set<MIROperand*>& callee_saved_regs,
                                     CodeGenContext& ctx, MIROperand* return_addr_reg) override {
        std::vector<std::pair<MIROperand*, MIROperand*>> saved;

        /* find the callee saved registers */
        /* allocate stack space for callee-saved registers */
        {
            for (auto op : callee_saved_regs) {
                auto size = getOperandSize(
                    ctx.registerInfo->getCanonicalizedRegisterType(op->type()));
                auto alignment = size;
                auto stack = func->add_stack_obj(ctx.next_id(), size, alignment, 0, StackObjectUsage::CalleeSaved);

                saved.emplace_back(op, stack);
            }
        }
        /* return address register */
        auto size = getOperandSize(return_addr_reg->type());
        auto alignment = size;
        auto stack = func->add_stack_obj(ctx.next_id(), size, alignment, 0, StackObjectUsage::CalleeSaved);
        saved.emplace_back(return_addr_reg, stack);


        /* insert the prologue and epilogue */
        {
            for (auto& block : func->blocks()) {
                auto& instructions = block->insts();

                // 1. 开始执行指令之前保存相关的调用者维护寄存器
                if (&block == &func->blocks().front()) {
                    for (auto [op, stack] : saved) {
                        auto inst = new MIRInst{InstStoreRegToStack};
                        inst->set_operand(0, stack);
                        inst->set_operand(1, op);
                        instructions.push_front(inst);
                    }
                }

                // 2. 函数返回之前将相关的调用者维护寄存器释放
                auto exit = instructions.back();
                auto& instInfo = ctx.instInfo.get_instinfo(exit);
                if (requireFlag(instInfo.inst_flag(), InstFlagReturn)) {
                    auto pos = std::prev(instructions.end());
                    for (auto [op, stack] : saved) {
                        auto inst = new MIRInst{InstLoadRegFromStack};
                        inst->set_operand(0, op);
                        inst->set_operand(1, stack);
                        instructions.insert(pos, inst);
                    }
                }
            }
        }
        return 0;
    }
public:  // alignment
    size_t get_stackpointer_alignment() override { return 8; }
};
/*
 * @brief: RISCVRegisterInfo Class
 * @note: RISC-V寄存器相关信息
 */
class RISCVRegisterInfo : public TargetRegisterInfo {
public:  // get function
    /* GPR(General Purpose Registers)/FPR(Floating Point Registers) */
    uint32_t get_alloca_class_cnt() { return 2; }
    uint32_t get_alloca_class(OperandType type) {
        switch (type) {
            case OperandType::Bool:
            case OperandType::Int8:
            case OperandType::Int16:
            case OperandType::Int32:
            case OperandType::Int64: return 0;
            case OperandType::Float32: return 1;
            default: assert(false && "invalid alloca class");
        }
    }

    std::vector<uint32_t>& get_allocation_list(uint32_t classId) {
        if (classId == 0) {  // General Purpose Registers
            static std::vector<uint32_t> list{
                // $a0-$a5
                RISCV::X10, RISCV::X11, RISCV::X12, RISCV::X13, RISCV::X14, RISCV::X15,
                // $a6-$a7
                RISCV::X16, RISCV::X17,
                // $t0-$t6
                RISCV::X5, RISCV::X6, RISCV::X7, RISCV::X28, RISCV::X29, RISCV::X30, RISCV::X31,
                // $s0-$s1
                RISCV::X8, RISCV::X9,
                // $s2-$s11
                RISCV::X18, RISCV::X19, RISCV::X20, RISCV::X21, RISCV::X22, RISCV::X23, RISCV::X24,
                RISCV::X25, RISCV::X26, RISCV::X27,
                // $gp
                RISCV::X3,
            };
            return list;
        } else if (classId == 1) {  // Floating Point Registers
            static std::vector<uint32_t> list{
                // $fa0-$fa5
                RISCV::F10, RISCV::F11, RISCV::F12, RISCV::F13, RISCV::F14, RISCV::F15,
                // $fa6-$fa7
                RISCV::F16, RISCV::F17,
                // $ft0-$ft11
                RISCV::F0, RISCV::F1, RISCV::F2, RISCV::F3, RISCV::F4, RISCV::F5, RISCV::F6, RISCV::F7,
                RISCV::F28, RISCV::F29, RISCV::F30, RISCV::F31,
                // $fs0-$fs1
                RISCV::F8, RISCV::F9,
                // $fs2-$fs11
                RISCV::F18, RISCV::F19, RISCV::F20, RISCV::F21, RISCV::F22, RISCV::F23, RISCV::F24,
                RISCV::F25, RISCV::F26, RISCV::F27,
            };
            return list;
        } else {
            assert(false && "invalid type regoster");
        }
    }
    OperandType getCanonicalizedRegisterTypeForClass(uint32_t classId) {
        return classId == 0 ? OperandType::Int32 : OperandType::Float32;
    }

    OperandType getCanonicalizedRegisterType(OperandType type) {
        switch (type) {
            case OperandType::Bool:
            case OperandType::Int8:
            case OperandType::Int16:
            case OperandType::Int32:
            case OperandType::Int64:
                return OperandType::Int64;
            case OperandType::Float32:
                return OperandType::Float32;
            default:
                assert(false && "valid operand type");
        }
    }
    OperandType getCanonicalizedRegisterType(uint32_t reg) {
        if(reg >= RISCV::GPRBegin and reg <= RISCV::GPREnd) {
            return OperandType::Int64;
        } else if(reg >= RISCV::FPRBegin and reg <= RISCV::FPREnd) {
            return OperandType::Float32;
        } else {
            assert(false && "valid operand type");
        }
    }
    MIROperand* get_return_address_register() { return RISCV::ra; }
    MIROperand* get_stack_pointer_register() { return RISCV::sp; }

public:  // check function
    bool is_legal_isa_reg_operand(MIROperand& op) {
        std::cerr << "Not Impl is_legal_isa_reg_operand" << std::endl;
        return false;
    }
    bool is_zero_reg(const uint32_t x) const { return x == RISCV::RISCVRegister::X0; }
};

/*
 * @brief: RISCVTarget Class
 * @note: RISC-V架构后端
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
    TargetInstInfo& get_target_inst_info() override { return RISCV::getRISCVInstInfo(); }
    TargetISelInfo& get_target_isel_info() override { return RISCV::getRISCVISelInfo(); }

    TargetScheduleModel& get_schedule_model() override {
        return RISCV::getRISCVScheduleModel();
    }
public:  // emit_assembly
    void postLegalizeFunc(MIRFunction& func, CodeGenContext& ctx) override;
    void emit_assembly(std::ostream& out, MIRModule& module) override;
};
}