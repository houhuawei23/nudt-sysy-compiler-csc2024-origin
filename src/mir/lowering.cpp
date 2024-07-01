#include "mir/mir.hpp"
#include "mir/lowering.hpp"
#include "mir/target.hpp"
#include "mir/iselinfo.hpp"

#include "mir/utils.hpp"
#include "mir/GraphColoringRegisterAllocation.hpp"
#include "mir/fastAllocator.hpp"
#include "mir/linearAllocator.hpp"

#include "target/riscv/riscvtarget.hpp"

namespace mir {
void create_mir_module(ir::Module& ir_module, MIRModule& mir_module, Target& target);
MIRFunction* create_mir_function(ir::Function* ir_func, MIRFunction* mir_func,
                                 CodeGenContext& codegen_ctx,
                                 LoweringContext& lowering_ctx);
MIRInst* create_mir_inst(ir::Instruction* ir_inst, LoweringContext& ctx);
void lower_GetElementPtr(ir::inst_iterator begin, ir::inst_iterator end, LoweringContext& ctx);

std::unique_ptr<MIRModule> create_mir_module(ir::Module& ir_module, Target& target) {
    auto mir_module_uptr = std::make_unique<MIRModule>(&ir_module, target);
    create_mir_module(ir_module, *mir_module_uptr, target);
    return mir_module_uptr;
}

union FloatUint32 {
    float f;
    uint32_t u;
} fu32;

void create_mir_module(ir::Module& ir_module, MIRModule& mir_module, Target& target) {
    constexpr bool debugLowering = false;

    auto& functions = mir_module.functions();
    auto& global_objs = mir_module.global_objs();
    LoweringContext lowering_ctx(mir_module, target);
    auto& func_map = lowering_ctx.func_map;
    auto& gvar_map = lowering_ctx.gvar_map;

    //! 1. for all functions, create MIRFunction
    for (auto func : ir_module.funcs()) {
        functions.push_back(std::make_unique<MIRFunction>(func->name(), &mir_module));
        func_map.emplace(func, functions.back().get());
    }

    //! 2. for all global variables, create MIRGlobalObject
    for (auto ir_gval : ir_module.gvalues()) {
        auto ir_gvar = dyn_cast<ir::GlobalVariable>(ir_gval);
        auto name = ir_gvar->name().substr(1);  // remove '@'
        auto type = dyn_cast<ir::PointerType>(ir_gvar->type())->base_type();
        size_t size = type->size();  // data size, not pointer size

        if (ir_gvar->is_init()) {  /* .data: 已初始化的、可修改的全局数据 */
            MIRDataStorage::Storage data;
            auto val = dyn_cast<ir::Constant>(ir_gvar->scalar_value());
            if (type->is_int()) {
                if (type->is_i32()) {
                    data.push_back(static_cast<uint32_t>(val->i32()));
                } else if (type->is_array()) {
                    // TODO: handle array init
                }
            } else if (type->is_float()) {
                /* NOTE: float to uint32_t, type cast, doesn't change the memory */
                fu32.f = val->f32(); data.push_back(fu32.u);
            }
            size_t align = 4;  // TODO: align
            auto mir_storage =
                std::make_unique<MIRDataStorage>(std::move(data), false, name);
            auto mir_gobj = std::make_unique<MIRGlobalObject>(
                align, std::move(mir_storage), &mir_module);
            mir_module.global_objs().push_back(std::move(mir_gobj));

            // mir_module.global_objs().push_back(
            //     std::make_unique<MIRGlobalObject>(
            //         align,
            //         std::make_unique<MIRDataStorage>(std::move(data), false,
            //                                          ir_gvar->name()),
            //         &mir_module));
        } else {  /* .bss: 未初始化的全局数据 */
            size_t align = 4;  // TODO: align
            auto mir_storage = std::make_unique<MIRZeroStorage>(size, name);
            auto mir_gobj = std::make_unique<MIRGlobalObject>(align, std::move(mir_storage), &mir_module);
            mir_module.global_objs().push_back(std::move(mir_gobj));
        }
        gvar_map.emplace(ir_gvar, mir_module.global_objs().back().get());
        gvar_map.emplace(ir_gvar, mir_module.global_objs().back().get());
    }

    // TODO: transformModuleBeforeCodeGen

    //! 3. codegen
    CodeGenContext codegen_ctx{target, target.get_datalayout(),
                               target.get_target_inst_info(),
                               target.get_target_frame_info(), MIRFlags{}};
    codegen_ctx.iselInfo = &target.get_target_isel_info();
    lowering_ctx._code_gen_ctx = &codegen_ctx;
    IPRAUsageCache infoIPRA;

    auto dumpStageWithMsg = [&](std::ostream& os, std::string_view stage, std::string_view msg) {
        enum class Style { RED, BOLD, RESET };

        static std::unordered_map<Style, std::string_view> styleMap = {
            {Style::RED, "\033[0;31m"},
            {Style::BOLD, "\033[1m"},
            {Style::RESET, "\033[0m"}};

        os << "\n";
        os << styleMap[Style::RED] << styleMap[Style::BOLD];
        os << "[" << stage << "] ";
        os << styleMap[Style::RESET];
        os << msg << std::endl;
    };

    //! 4. lower all functions
    for (auto& ir_func : ir_module.funcs()) {
        auto mir_func = func_map[ir_func];
        if (ir_func->blocks().empty()) continue;
        if (debugLowering) {
            dumpStageWithMsg(std::cerr, "Before Lowering", mir_func->name());
            ir_func->print(std::cerr);
        }

        /* 4.1: lower function body to generic MIR */
        {
            create_mir_function(ir_func, mir_func, codegen_ctx, lowering_ctx);
            if (debugLowering) {
                dumpStageWithMsg(std::cerr, "After Lowering", mir_func->name());
                mir_func->print(std::cerr, codegen_ctx);
            }
        }

        /* 4.2: instruction selection */
        {
            ISelContext isel_ctx(codegen_ctx);
            isel_ctx.run_isel(mir_func);
            if (debugLowering) {
                dumpStageWithMsg(std::cerr, "After ISEL", mir_func->name());
                mir_func->print(std::cerr, codegen_ctx);
            }
        }
        /* 4.3 register coalescing */

        /* 4.4 peephole optimization */

        /* 4.5 pre-RA legalization */

        /* 4.6 pre-RA scheduling, minimize register usage */
        // preRASchedule(*mir_func, codegen_ctx);

        /* 4.7 register allocation */
        {
            codegen_ctx.registerInfo = new RISCVRegisterInfo();
            if (codegen_ctx.registerInfo) {
                GraphColoringAllocate(*mir_func, codegen_ctx, infoIPRA);

                if (debugLowering) {
                    dumpStageWithMsg(std::cerr, "After RA", mir_func->name());
                    mir_func->print(std::cerr, codegen_ctx);
                }
            }
        }

        /* 4.8 stack allocation */
        if (codegen_ctx.registerInfo) {
            /* after sa, all stack objects are allocated with .offset */
            allocateStackObjects(mir_func, codegen_ctx);
            codegen_ctx.flags.postSA = true;
            if (debugLowering) {
                dumpStageWithMsg(std::cerr, "After SA", mir_func->name());
                mir_func->print(std::cerr, codegen_ctx);
            }
        }

        /* 4.9 post-RA scheduling, minimize cycles */

        /* 4.10 post legalization */
        postLegalizeFunc(*mir_func, codegen_ctx);
        /* 4.11 verify */

        if (debugLowering) {
            dumpStageWithMsg(std::cerr, "Finished CodeGen", mir_func->name());
            mir_func->print(std::cerr, codegen_ctx);
        }
    }
    /* module verify */
}

MIRFunction* create_mir_function(ir::Function* ir_func, MIRFunction* mir_func,
                                 CodeGenContext& codegen_ctx, LoweringContext& lowering_ctx) {
    constexpr bool debugLowerFunc = false;
    // TODO: before lowering, get some analysis pass result
    /* aligenment */
    /* range */
    /* dom */

    //! 1. map from ir to mir
    auto& block_map = lowering_ctx._block_map;
    std::unordered_map<ir::Value*, MIROperand*> storage_map;
    auto& target = codegen_ctx.target;
    auto& datalayout = target.get_datalayout();

    for (auto ir_block : ir_func->blocks()) {
        mir_func->blocks().push_back(std::make_unique<MIRBlock>(mir_func, "label" + std::to_string(codegen_ctx.next_id_label())));
        block_map.emplace(ir_block, mir_func->blocks().back().get());
    }

    //! 2. emitPrologue for function
    {
        for (auto ir_arg : ir_func->args()) {  // assign vreg to arg
            auto vreg = lowering_ctx.new_vreg(ir_arg->type());
            lowering_ctx.add_valmap(ir_arg, vreg);
            mir_func->args().push_back(vreg);
        }
        lowering_ctx.set_mir_block(block_map.at(ir_func->entry()));  // entry
        // TODO: implement riscv frameinfo.emit_prologue()
        codegen_ctx.frameInfo.emit_prologue(mir_func, lowering_ctx);
    }

    //! 3. Process Alloca Instruction, New Stack Object for Each Alloca Instruction
    {
        lowering_ctx.set_mir_block(block_map.at(ir_func->entry()));  // NOTE: all alloca in entry
        for (auto& ir_inst : ir_func->entry()->insts()) {
            if (ir_inst->scid() != ir::Value::vALLOCA) continue;

            auto pointee_type = dyn_cast<ir::PointerType>(ir_inst->type())->base_type();
            uint32_t align = 4;  // TODO: align, need bind to ir object
            /* 3.1 allocate stack storage */
            auto storage = mir_func->add_stack_obj(codegen_ctx.next_id(),
                                                   static_cast<uint32_t>(pointee_type->size()),
                                                   align, 0, StackObjectUsage::Local);
            storage_map.emplace(ir_inst, storage);
            /* 3.2 emit load stack object address instruction */
            auto addr = lowering_ctx.new_vreg(lowering_ctx.get_ptr_type());
            auto ldsa_inst = new MIRInst{InstLoadStackObjectAddr};
            ldsa_inst->set_operand(0, addr);
            ldsa_inst->set_operand(1, storage);
            lowering_ctx.emit_inst(ldsa_inst);
            /* 3.3 map */
            lowering_ctx.add_valmap(ir_inst, addr);
        }
    }

    //! 4. Lowering All Blocks
    {
        for (auto ir_block : ir_func->blocks()) {
            auto mir_block = block_map[ir_block];
            lowering_ctx.set_mir_block(mir_block);
            auto& instructions = ir_block->insts();
            for (auto iter = instructions.begin(); iter != instructions.end();) {
                auto ir_inst = *iter;
                if (ir_inst->scid() == ir::Value::vALLOCA) iter++;
                else if (ir_inst->scid() == ir::Value::vGETELEMENTPTR) {
                    auto end = iter; end++;
                    while (end != instructions.end() && (*end)->scid() == ir::Value::vGETELEMENTPTR) {
                        auto preInst = std::prev(end);
                        if (dyn_cast<ir::GetElementPtrInst>(*end)->get_value() == (*preInst)) end++;
                        else break;
                    }
                    lower_GetElementPtr(iter, end, lowering_ctx);
                    iter = end;
                } else {
                    create_mir_inst(ir_inst, lowering_ctx);
                    iter++;
                }
                if (debugLowerFunc) {
                    ir_inst->print(std::cerr);
                    std::cerr << std::endl;
                }
            }
        }
    }
    return mir_func;
}
void lower(ir::UnaryInst* ir_inst, LoweringContext& ctx);
void lower(ir::BinaryInst* ir_inst, LoweringContext& ctx);
void lower(ir::BranchInst* ir_inst, LoweringContext& ctx);
void lower(ir::LoadInst* ir_inst, LoweringContext& ctx);
void lower(ir::StoreInst* ir_inst, LoweringContext& ctx);
void lower(ir::ICmpInst* ir_inst, LoweringContext& ctx);
void lower(ir::FCmpInst* ir_inst, LoweringContext& ctx);
void lower(ir::CallInst* ir_inst, LoweringContext& ctx);
void lower(ir::ReturnInst* ir_inst, LoweringContext& ctx);
void lower(ir::BranchInst* ir_inst, LoweringContext& ctx);

MIRInst* create_mir_inst(ir::Instruction* ir_inst, LoweringContext& ctx) {
    switch (ir_inst->scid()) {
        case ir::Value::vFNEG:
        case ir::Value::vTRUNC:
        case ir::Value::vZEXT:
        case ir::Value::vSEXT:
        case ir::Value::vFPTRUNC:
        case ir::Value::vFPTOSI:
        case ir::Value::vSITOFP:
            lower(dyn_cast<ir::UnaryInst>(ir_inst), ctx);
            break;
        case ir::Value::vADD:
        case ir::Value::vFADD:
        case ir::Value::vSUB:
        case ir::Value::vFSUB:
        case ir::Value::vMUL:
        case ir::Value::vFMUL:
        case ir::Value::vUDIV:
        case ir::Value::vSDIV:
        case ir::Value::vFDIV:
        case ir::Value::vUREM:
        case ir::Value::vSREM:
        case ir::Value::vFREM:
            lower(dyn_cast<ir::BinaryInst>(ir_inst), ctx);
            break;
        case ir::Value::vIEQ:
        case ir::Value::vINE:
        case ir::Value::vISGT:
        case ir::Value::vISGE:
        case ir::Value::vISLT:
        case ir::Value::vISLE:
            lower(dyn_cast<ir::ICmpInst>(ir_inst), ctx);
            break;
        case ir::Value::vFOEQ:
        case ir::Value::vFONE:
        case ir::Value::vFOGT:
        case ir::Value::vFOGE:
        case ir::Value::vFOLT:
        case ir::Value::vFOLE:
            lower(dyn_cast<ir::FCmpInst>(ir_inst), ctx);
            break;
        case ir::Value::vALLOCA:
            break;
        case ir::Value::vLOAD:
            lower(dyn_cast<ir::LoadInst>(ir_inst), ctx);
            break;
        case ir::Value::vSTORE:
            lower(dyn_cast<ir::StoreInst>(ir_inst), ctx);
            break;
        case ir::Value::vRETURN:
            lower(dyn_cast<ir::ReturnInst>(ir_inst), ctx);
            break;
        case ir::Value::vBR:
            lower(dyn_cast<ir::BranchInst>(ir_inst), ctx);
            break;
        case ir::Value::vCALL:
            lower(dyn_cast<ir::CallInst>(ir_inst), ctx);
            break;
        case ir::Value::vGETELEMENTPTR:
        default:
            assert(false && "not supported inst");
    }
    return nullptr;
}
void lower(ir::UnaryInst* ir_inst, LoweringContext& ctx) {
    auto gc_instid = [scid = ir_inst->scid()] {
        switch (scid) {
            case ir::Value::vFNEG:
                return InstFNeg;
            case ir::Value::vTRUNC:
                return InstTrunc;
            case ir::Value::vZEXT:
                return InstZExt;
            case ir::Value::vSEXT:
                return InstSExt;
            case ir::Value::vFPTRUNC:
                assert(false && "not supported unary inst");
                // return InstFPTrunc;
            case ir::Value::vFPTOSI:
                return InstF2S;
            case ir::Value::vSITOFP:
                return InstS2F;
            default:
                assert(false && "not supported unary inst");
        }
    }();

    auto ret = ctx.new_vreg(ir_inst->type());
    auto inst = new MIRInst(gc_instid);
    inst->set_operand(0, ret);
    inst->set_operand(1, ctx.map2operand(ir_inst->operand(0)));
    ctx.emit_inst(inst);
    ctx.add_valmap(ir_inst, ret);
}

/**
 * IR:
 * `<result> = icmp <cond> <ty> <op1>, <op2>`
 * yields i1 or <N x i1>:result
 *
 * MIRGeneric:
 * `ICmp dst, src1, src2, op`
 */
void lower(ir::ICmpInst* ir_inst, LoweringContext& ctx) {
    // TODO: Need Test
    auto op = [scid = ir_inst->scid()] {
        switch (scid) {
            case ir::Value::vIEQ:
                return CompareOp::ICmpEqual;
            case ir::Value::vINE:
                return CompareOp::ICmpNotEqual;
            case ir::Value::vISGT:
                return CompareOp::ICmpSignedGreaterThan;
            case ir::Value::vISGE:
                return CompareOp::ICmpSignedGreaterEqual;
            case ir::Value::vISLT:
                return CompareOp::ICmpSignedLessThan;
            case ir::Value::vISLE:
                return CompareOp::ICmpSignedLessEqual;
            default:
                assert(false && "not supported icmp inst");
        }
    }();

    auto ret = ctx.new_vreg(ir_inst->type());
    auto inst = new MIRInst(InstICmp);
    inst->set_operand(0, ret);
    inst->set_operand(1, ctx.map2operand(ir_inst->operand(0)));
    inst->set_operand(2, ctx.map2operand(ir_inst->operand(1)));
    // TODO: set condition code: eq, ne, sgt, sge, slt, sle
    inst->set_operand(3, MIROperand::as_imm(static_cast<uint32_t>(op), OperandType::Special));
    ctx.emit_inst(inst);
    ctx.add_valmap(ir_inst, ret);
}
/**
 * IR:
 * `<result> = fcmp [fast-math flags]* <cond> <ty> <op1>, <op2>`
 * yields i1 or <N x i1>:result
 *
 * MIRGeneric:
 * `FCmp dst, src1, src2, op`
 */
void lower(ir::FCmpInst* ir_inst, LoweringContext& ctx) {
    // TODO: Need Test
    auto op = [scid = ir_inst->scid()] {
        switch (scid) {
            case ir::Value::vFOEQ:
                return CompareOp::FCmpOrderedEqual;
            case ir::Value::vFONE:
                return CompareOp::FCmpOrderedNotEqual;
            case ir::Value::vFOGT:
                return CompareOp::FCmpOrderedGreaterThan;
            case ir::Value::vFOGE:
                return CompareOp::FCmpOrderedGreaterEqual;
            case ir::Value::vFOLT:
                return CompareOp::FCmpOrderedLessThan;
            case ir::Value::vFOLE:
                return CompareOp::FCmpOrderedLessEqual;
            default:
                assert(false && "not supported fcmp inst");
        }
    }();

    auto ret = ctx.new_vreg(ir_inst->type());
    auto inst = new MIRInst(InstFCmp);
    inst->set_operand(0, ret);
    inst->set_operand(1, ctx.map2operand(ir_inst->operand(0)));
    inst->set_operand(2, ctx.map2operand(ir_inst->operand(1)));
    // TODO: set condition code: oeq, one, ogt, oge, olt, ole
    inst->set_operand(
        3, MIROperand::as_imm(static_cast<uint32_t>(op), OperandType::Special));
    ctx.emit_inst(inst);
    ctx.add_valmap(ir_inst, ret);
}

/* CallInst */
void lower(ir::CallInst* ir_inst, LoweringContext& ctx) {
    // TODO: Need Test
    ctx._target.get_target_frame_info().emit_call(ir_inst, ctx);
}

/* BinaryInst */
void lower(ir::BinaryInst* ir_inst, LoweringContext& ctx) {
    auto gc_instid = [scid = ir_inst->scid()] {
        switch (scid) {
            case ir::Value::vADD: return InstAdd;
            case ir::Value::vFADD: return InstFAdd;
            case ir::Value::vSUB: return InstSub;
            case ir::Value::vFSUB: return InstFSub;
            case ir::Value::vMUL: return InstMul;
            case ir::Value::vFMUL: return InstFMul;
            case ir::Value::vUDIV: return InstUDiv;
            case ir::Value::vSDIV: return InstSDiv;
            case ir::Value::vFDIV: return InstFDiv;
            case ir::Value::vUREM: return InstURem;
            case ir::Value::vSREM: return InstSRem;
            default: assert(false && "not supported binary inst");
        }
    }();

    auto ret = ctx.new_vreg(ir_inst->type());
    auto inst = new MIRInst(gc_instid);
    inst->set_operand(0, ret);
    inst->set_operand(1, ctx.map2operand(ir_inst->operand(0)));
    inst->set_operand(2, ctx.map2operand(ir_inst->operand(1)));
    ctx.emit_inst(inst);
    ctx.add_valmap(ir_inst, ret);
}

/* BranchInst */
void emit_branch(ir::BasicBlock* srcblock, ir::BasicBlock* dstblock, LoweringContext& lctx);
void lower(ir::BranchInst* ir_inst, LoweringContext& ctx) {
    auto src_block = ir_inst->parent();

    if (ir_inst->is_cond()) {
        // TODO: conditional branch
        // branch
        auto inst = new MIRInst(InstBranch);
        inst->set_operand(0, ctx.map2operand(ir_inst->cond()));
        inst->set_operand(
            1, MIROperand::as_reloc(ctx.map2block(ir_inst->iftrue())));
        inst->set_operand(2, MIROperand::as_prob(0.5));
        ctx.emit_inst(inst);
        /* fallthrough, jump */
        auto jump_inst = new MIRInst(InstJump);
        jump_inst->set_operand(0, MIROperand::as_reloc(ctx.map2block(ir_inst->iffalse())));
        ctx.emit_inst(jump_inst);

    } else {  // unconditional branch
        auto dst_block = ir_inst->dest();
        emit_branch(src_block, dst_block, ctx);
    }
}
void emit_branch(ir::BasicBlock* srcblock, ir::BasicBlock* dstblock, LoweringContext& lctx) {
    auto dst_mblock = lctx.map2block(dstblock);
    auto src_mblock = lctx.map2block(srcblock);
    auto operand = MIROperand::as_reloc(dst_mblock);
    auto inst = new MIRInst(InstJump);
    inst->set_operand(0, operand);
    lctx.emit_inst(inst);
}

/* LoadInst */
void lower(ir::LoadInst* ir_inst, LoweringContext& ctx) {
    auto ret = ctx.new_vreg(ir_inst->type()); auto ptr = ctx.map2operand(ir_inst->operand(0));
    assert(ret != nullptr && ptr != nullptr);

    uint32_t align = 4;
    auto inst = new MIRInst(InstLoad);
    inst->set_operand(0, ret);
    inst->set_operand(1, ptr);
    inst->set_operand(2, MIROperand::as_imm(align, OperandType::Special));
    ctx.emit_inst(inst);
    ctx.add_valmap(ir_inst, ret);
}

/* StoreInst */
void lower(ir::StoreInst* ir_inst, LoweringContext& ctx) {
    auto inst = new MIRInst(InstStore);
    inst->set_operand(0, ctx.map2operand(ir_inst->ptr()));
    inst->set_operand(1, ctx.map2operand(ir_inst->value()));
    uint32_t align = 4;
    inst->set_operand(2, MIROperand::as_imm(align, OperandType::Special));
    ctx.emit_inst(inst);
}

/* ReturnInst */
void lower(ir::ReturnInst* ir_inst, LoweringContext& ctx) {
    ctx._target.get_target_frame_info().emit_return(ir_inst, ctx);
}

/*
 * @brief: lower GetElementPtrInst [begin, end)
 * @note: 
 *      1. Array: <result> = getelementptr <type>, <type>* <ptrval>, i32 0, i32 <idx>
 *      2. Pointer: <result> = getelementptr <type>, <type>* <ptrval>, i32 <idx>
 */
void lower_GetElementPtr(ir::inst_iterator begin, ir::inst_iterator end, LoweringContext& ctx) {
    const auto dims = dyn_cast<ir::GetElementPtrInst>(*begin)->cur_dims();
    const auto dims_cnt = dyn_cast<ir::GetElementPtrInst>(*begin)->cur_dims_cnt();
    const int id = dyn_cast<ir::GetElementPtrInst>(*begin)->get_id();
    const auto base = ctx.map2operand(dyn_cast<ir::GetElementPtrInst>(*begin)->get_value());  // 基地址

    auto iter = begin; ir::Value* instEnd = nullptr; MIROperand* ptr = base;
    std::vector<MIROperand*> idx;
    while (iter != end) {
        idx.push_back(ctx.map2operand(dyn_cast<ir::GetElementPtrInst>(*iter)->get_index()));
        instEnd = *iter;
        iter++;
    }

    if (id == 0) {  // 指针

    } else if (id == 1) {  // 高维数组
        for (int dimension = dims_cnt - 1; dimension > 0; dimension--) {

        }
    } else {  // 一维数组
        /* 计算偏移量: 基地址 + 偏移量 (base + offset) */
        const auto offset = idx[0];
        auto newPtr = ctx.new_vreg(OperandType::Int64);
        auto newInst = new MIRInst(InstAdd);
        newInst->set_operand(0, newPtr); newInst->set_operand(1, ptr); newInst->set_operand(2, offset);
        ctx.emit_inst(newInst);
        ptr = newPtr;
    }
    ctx.add_valmap(instEnd, ptr);
}
}  // namespace mir
