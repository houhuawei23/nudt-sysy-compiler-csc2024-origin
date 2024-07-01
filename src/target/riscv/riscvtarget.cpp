#include "mir/utils.hpp"
#include "target/riscv/riscvtarget.hpp"
#include "autogen/riscv/InstInfoDecl.hpp"
#include "autogen/riscv/ISelInfoDecl.hpp"

namespace mir {
constexpr int32_t passingByRegBase = 0x100000;
void RISCVFrameInfo::emit_call(ir::CallInst* inst, LoweringContext& lowering_ctx) {
    bool isDebug = false;

    auto irCalleeFunc = inst->callee(); /* Function* */
    auto mirCalleeFunc = lowering_ctx.func_map.at(irCalleeFunc);
    assert(mirCalleeFunc);
    // auto
    /* calculate */
    int32_t curOffset = 0;
    std::vector<int32_t> offsets;

    int32_t gprCount = 0, fprCount = 0;

    /** 计算参数的偏移量，确定哪些参数通过寄存器传递，哪些通过栈传递。 */
    for (auto use : inst->rargs()) {
        auto arg = use->value();
        if (not arg->type()->is_float()) {
            if (gprCount < 8) {  // TODO: max count should in RISCVRegisterInfo
                std::cerr << "gprCount: " << gprCount << std::endl;
                offsets.push_back(passingByRegBase + gprCount++);
                continue;
            }
        } else {
            if (fprCount < 8) {
                std::cerr << "fprCount: " << fprCount << std::endl;
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

    auto mirFunc = lowering_ctx.get_mir_block()->parent();
    /** TODO: 为通过栈传递的参数分配栈空间，并生成相应的存储指令。 */
    for (uint32_t idx = 0; idx < inst->rargs().size(); idx++) {
        auto arg = inst->operand(idx);
        const auto offset = offsets[idx];
        auto val = lowering_ctx.map2operand(arg);
        auto size = arg->type()->size();
        auto align = 4;  // TODO: check alignment

        if (offset < passingByRegBase) {
            /* pass by stack */
            auto obj = mirFunc->add_stack_obj(
                lowering_ctx._code_gen_ctx->next_id(), size, align, offset,
                StackObjectUsage::CalleeArgument);
            if (not isOperandVRegOrISAReg(val)) {
                auto reg = lowering_ctx.new_vreg(val->type());
                lowering_ctx.emit_copy(reg, val);
                val = reg;
            }
            auto newInst = new MIRInst{InstStoreRegToStack};
            newInst->set_operand(0, obj);
            newInst->set_operand(1, val);
            lowering_ctx.emit_inst(newInst);
        }
    }
    /** TODO: 为通过寄存器传递的参数生成相应的寄存器赋值指令。 */
    for (uint32_t idx = 0; idx < inst->rargs().size(); idx++) {
        auto arg = inst->operand(idx);
        const auto offset = offsets[idx];
        auto val = lowering_ctx.map2operand(arg);
        if (offset >= passingByRegBase) {
            /* pass by reg */
            MIROperand* dst = nullptr;
            if (isFloatType(val->type())) {
                dst = MIROperand::as_isareg(
                    RISCV::F10 +
                        static_cast<uint32_t>(offset - passingByRegBase),
                    OperandType::Float32);

            } else {
                /* int, or else */
                std::cerr << "X" << (offset - passingByRegBase) << std::endl;
                dst = MIROperand::as_isareg(
                    RISCV::X10 +
                        static_cast<uint32_t>(offset - passingByRegBase),
                    OperandType::Int64);
            }
            assert(dst);
            lowering_ctx.emit_copy(dst, val);
        }
    }
    /** TODO: 生成跳转至被调用函数的指令。*/
    auto callInst = new MIRInst(RISCV::JAL);
    callInst->set_operand(0, MIROperand::as_reloc(mirCalleeFunc));
    lowering_ctx.emit_inst(callInst);
    const auto irRetType = inst->callee()->ret_type();
    /** TODO:
     * 如果函数返回值不是 void，生成代码以将返回值存储到虚拟寄存器中，
     * 并将其添加到调用指令的o perand 列表中 */
    if (irRetType->is_void()) {
        return;
    }
    /* not void */
    auto retReg = lowering_ctx.new_vreg(irRetType);
    MIROperand* val = nullptr;
    if (irRetType->is_float()) {
        /* return by $fa0 */
        val = MIROperand::as_isareg(RISCV::F10, OperandType::Float32);
    } else {
        /* return by $a0 */
        val = MIROperand::as_isareg(RISCV::X10, OperandType::Int64);
    }
    lowering_ctx.emit_copy(retReg, val);
    lowering_ctx.add_valmap(inst, retReg);
}

void RISCVFrameInfo::emit_prologue(MIRFunction* func,
                                   LoweringContext& lowering_ctx) {
    // TODO: implement prologue
    const auto& args = func->args();
    int32_t curOffset = 0;
    /* off >= passingByGPR: passing by reg[off - passingByRegBase] */
    std::vector<int32_t> offsets;
    int32_t gprCount = 0, fprCount = 0;

    /* traverse args, split into reg/stack */
    for (auto& arg : args) {
        if (isIntType(arg->type())) {
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

        /* pass by stack */
        int32_t size = getOperandSize(arg->type());
        int32_t align = 4;  // TODO: check alignment
        int32_t minimumSize = sizeof(uint64_t);
        size = std::max(size, minimumSize);
        align = std::max(align, minimumSize);
        curOffset = (curOffset + align - 1) / align * align;
        offsets.push_back(curOffset);
        curOffset += size;
    }
    /* pass by register */
    for (uint32_t idx = 0; idx < args.size(); idx++) {
        const auto offset = offsets[idx];
        const auto& arg = args[idx];
        if (offset >= passingByRegBase) {
            /* $a0-$a7, $f0-$f7 */
            MIROperand* src = nullptr;
            if (isFloatType(arg->type())) {
                src = MIROperand::as_isareg(
                    RISCV::F10 +
                        static_cast<uint32_t>(offset - passingByRegBase),
                    OperandType::Float32);
            } else {
                src = MIROperand::as_isareg(
                    RISCV::X10 +
                        static_cast<uint32_t>(offset - passingByRegBase),
                    OperandType::Int64);
            }
            assert(src);
            lowering_ctx.emit_copy(arg, src);
        }
    }
    /* pass by stack */
    for (uint32_t idx = 0; idx < args.size(); idx++) {
        const auto offset = offsets[idx];
        const auto& arg = args[idx];
        const auto size = getOperandSize(arg->type());
        const auto align = 4;  // TODO: check alignment
        if (offset < passingByRegBase) {
            auto obj =
                func->add_stack_obj(lowering_ctx._code_gen_ctx->next_id(), size,
                                    align, offset, StackObjectUsage::Argument);
            auto newInst = new MIRInst{InstLoadRegFromStack};
            newInst->set_operand(0, arg);
            newInst->set_operand(1, obj);
            lowering_ctx.emit_inst(newInst);
        }
    }
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

void RISCVTarget::postLegalizeFunc(MIRFunction& func, CodeGenContext& ctx) {
    constexpr bool isDebug = false;

    auto dumpInst = [&](MIRInst* inst) {
        auto& instInfo = ctx.instInfo.get_instinfo(inst);
        instInfo.print(std::cerr << "rvPostLegalizeFunc: ", *inst, false);
        std::cerr << std::endl;
    };
    if (isDebug)
        func.print(std::cerr, ctx);
    /* fix pcrel addressing */
    for (auto blockIter = func.blocks().begin();
         blockIter != func.blocks().end();) {
        if (isDebug)
            std::cerr << "block: " << blockIter->get()->name() << std::endl;
        auto nextIter = std::next(blockIter);

        /* origin reloc -> (dst -> block)*/
        std::unordered_map<MIRRelocable*,
                           std::unordered_map<MIROperand*, MIRBlock*>>
            auipcMap;
        while (true) {
            auto& insts = blockIter->get()->insts();
            if (insts.empty()) {
                break;
            }
            bool isNewBlock = false;
            for (auto instIter = insts.begin(); instIter != insts.end();
                 instIter++) {
                auto& inst = *instIter;
                if (isDebug) {
                    dumpInst(inst);
                }
                if (inst->opcode() == RISCV::AUIPC) {
                    /* AUIPC dst, imm */
                    if (instIter == insts.begin() &&
                        blockIter != func.blocks().begin()) {
                        if (isDebug)
                            std::cerr << "first in block" << std::endl;
                        assert(inst->operand(1)->type() ==
                               OperandType::HighBits);
                        /** first inst in block, block label lowBits is just
                         * inst's dst lowBits */
                        // auipcMap[inst->operand(1)][inst->operand(0)] =
                        //     getLowBits(MIROperand::as_reloc(blockIter->get()));
                        // auto t = blockIter->get();
                        auipcMap[inst->operand(1)->reloc()][inst->operand(0)] =
                            blockIter->get();
                    } else {
                        if (isDebug)
                            std::cerr << "not first in block" << std::endl;
                        /** other insts */
                        auto newBlock = std::make_unique<MIRBlock>(
                            &func,
                            "pcrel" + std::to_string(ctx.next_id_label()));
                        auto& newInsts = newBlock->insts();
                        newInsts.splice(newInsts.begin(), insts, instIter,
                                        insts.end());
                        blockIter =
                            func.blocks().insert(nextIter, std::move(newBlock));
                        isNewBlock = true;
                        break; /* break instIter for insts */
                    }
                } else {
                    /* not AUIPC */
                    auto& instInfo = ctx.instInfo.get_instinfo(inst);

                    // instInfo.print(std::cerr << "!!", *inst,  false);
                    for (uint32_t idx = 0; idx < instInfo.operand_num();
                         idx++) {
                        auto operand = inst->operand(idx);

                        auto getBase = [&] {
                            switch (inst->opcode()) {
                                case RISCV::ADDI: {
                                    /* ADDI dst, src, imm */
                                    return inst->operand(1);
                                }
                                default: {
                                    /* must be load or store */
                                    assert(requireOneFlag(
                                        instInfo.inst_flag(),
                                        InstFlagLoad | InstFlagStore));
                                    /* load dst, imm(src)
                                     * store src2, imm(src1) */
                                    return inst->operand(2);
                                }
                            }
                        };

                        if (operand->is_reloc() &&
                            operand->type() == OperandType::LowBits) {
                            auto pcrelBlock =
                                auipcMap.at(operand->reloc()).at(getBase());
                            auto op =
                                getLowBits(MIROperand::as_reloc(pcrelBlock));
                            inst->set_operand(idx, op);
                            if (isDebug) {
                                instInfo.print(
                                    std::cerr << "fix pcrel: ", *inst, false);
                            }
                        }
                    }
                }
            }
            if (not isNewBlock)
                break;
        }  // while

        blockIter = nextIter;
    }  // blockIter
    if (isDebug) func.print(std::cerr, ctx);
}  // postLegalizeFunc

void RISCVTarget::emit_assembly(std::ostream& out, MIRModule& module) {
    auto& target = *this;
    CodeGenContext codegen_ctx{
        target, target.get_datalayout(), target.get_target_inst_info(),
        target.get_target_frame_info(), MIRFlags{false, false}};
    dump_assembly(out, module, codegen_ctx);
}

}  // namespace mir