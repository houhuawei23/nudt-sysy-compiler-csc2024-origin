#include "mir/utils.hpp"

#include "target/riscv/riscvtarget.hpp"
#include "autogen/riscv/InstInfoDecl.hpp"
#include "autogen/riscv/ISelInfoDecl.hpp"

namespace mir {
constexpr int32_t passingByRegBase = 0x100000;

void RISCVFrameInfo::emit_call(ir::CallInst* inst,
                               LoweringContext& lowering_ctx) {
    /* calculate */
    int32_t curOffset = 0;
    std::vector<int32_t> offsets;

    int32_t gprCount = 0, fprCount = 0;

    /** 计算参数的偏移量，确定哪些参数通过寄存器传递，哪些通过栈传递。 */
    for (auto use : inst->rargs()) {
        auto arg = use->value();
        if (not arg->type()->is_float()) {
            if (gprCount < 8) {
                offsets.push_back(passingByRegBase + gprCount++);
                continue;
            }
        } else {
            if (fprCount < 8) {
                offsets.push_back(passingByRegBase + fprCount++);
                continue;
            }
        }
        /* 如果寄存器已满，则计算栈上的位置，更新当前栈偏移curOffset */
        int32_t size = arg->type()->size();
        int32_t align = 4;  // TODO: check alignment

        int32_t minimumSize = sizeof(uint64_t);
        size = std::max(size, minimumSize);
        align = std::max(align, minimumSize);
        /*
        1. `curOffset + alignment - 1`：
        首先，将当前的栈偏移`curOffset`与参数的对齐字节数`alignment`相加，
        然后减去1。这样做的目的是为了确保即使在加上当前参数的大小后，栈地址仍然在对齐边界上。
        2. `(curOffset + alignment - 1) / alignment`：
        接着，将上一步的结果除以`alignment`。
        这个操作会得到一个小于或等于原栈偏移的值，且这个值是`alignment`的整数倍。
        这意味着，无论当前栈偏移是多少，除以对齐字节数后，都会得到一个对齐的地址。
        3. `/ alignment * alignment`：
        最后，将上一步的结果再乘以`alignment`。
        这样做是为了确保栈偏移量是`alignment`的整数倍，满足参数的对齐要求。
        */
        curOffset = (curOffset + align - 1) / align * align;
        offsets.push_back(curOffset);
        curOffset += size;
    }

    /** TODO: 为通过栈传递的参数分配栈空间，并生成相应的存储指令。 */

    /** TODO: 为通过寄存器传递的参数生成相应的寄存器赋值指令。 */

    /** TODO: 生成跳转至被调用函数的指令。*/

    /** TODO:
     * 如果函数返回值不是 void，生成代码以将返回值存储到虚拟寄存器中，
     * 并将其添加到调用指令的o perand 列表中 */
}

void RISCVFrameInfo::emit_prologue(MIRFunction* func,
                                   LoweringContext& lowering_ctx) {
    // TODO: implement prologue
    std::cerr << "RISCV prologue not implemented" << std::endl;
}

void RISCVFrameInfo::emit_return(ir::ReturnInst* ir_inst,
                                 LoweringContext& lowering_ctx) {
    // TODO: implement emit return
    if (not ir_inst->operands().empty()) {  // has return value
        // TODO
        auto retval = ir_inst->return_value();
        if (retval->type()->is_float()) {
            /* return by $fa0 */
            lowering_ctx.emit_copy(
                MIROperand::as_isareg(RISCV::F10, OperandType::Float32),
                lowering_ctx.map2operand(retval));
        } else if (retval->type()->is_int()) {
            /* return by $a0 */
            lowering_ctx.emit_copy(
                MIROperand::as_isareg(RISCV::X10, OperandType::Int64),
                lowering_ctx.map2operand(retval));
        }
    }
    auto inst = new MIRInst(RISCV::RET);
    lowering_ctx.emit_inst(inst);
}

void RISCVTarget::emit_assembly(std::ostream& out, MIRModule& module) {
    auto& target = *this;
    CodeGenContext codegen_ctx{
        target, target.get_datalayout(), target.get_target_inst_info(),
        target.get_target_frame_info(), MIRFlags{false, false}};
    dump_assembly(out, module, codegen_ctx);
}

}  // namespace mir