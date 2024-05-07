#include "mir/mir.hpp"
#include "mir/lowering.hpp"
#include "mir/target.hpp"
#include "mir/iselinfo.hpp"

#include "mir/utils.hpp"
#include "mir/fastAllocator.hpp"
#include "mir/linearAllocator.hpp"

#include "target/riscv/riscvtarget.hpp"
namespace mir {
//! declare
void create_mir_module(ir::Module& ir_module,
                       MIRModule& mir_module,
                       Target& target);

MIRFunction* create_mir_function(ir::Function* ir_func,
                                 MIRFunction* mir_func,
                                 CodeGenContext& codegen_ctx,
                                 LoweringContext& lowering_ctx);

MIRInst* create_mir_inst(ir::Instruction* ir_inst, LoweringContext& ctx);

//! implementation
std::unique_ptr<MIRModule> create_mir_module(ir::Module& ir_module,
                                             Target& target) {
    auto mir_module_uptr = std::make_unique<MIRModule>(&ir_module, target);
    create_mir_module(ir_module, *mir_module_uptr, target);
    return mir_module_uptr;
}

union FloatUint32 {
    float f;
    uint32_t u;
} fu32;

void create_mir_module(ir::Module& ir_module,
                       MIRModule& mir_module,
                       Target& target) {
    auto& functions = mir_module.functions();
    auto& global_objs = mir_module.global_objs();

    LoweringContext lowering_ctx(mir_module, target);

    auto& func_map = lowering_ctx.func_map;
    auto& gvar_map = lowering_ctx.gvar_map;

    //! 1. for all functions, create MIRFunction
    for (auto func : ir_module.funcs()) {
        functions.push_back(
            std::make_unique<MIRFunction>(func->name(), &mir_module));
        func_map.emplace(func, functions.back().get());
    }

    //! 2. for all global variables, create MIRGlobalObject
    for (auto ir_gval : ir_module.gvalues()) {
        auto ir_gvar = dyn_cast<ir::GlobalVariable>(ir_gval);
        auto type = dyn_cast<ir::PointerType>(ir_gvar->type())->base_type();
        size_t size = type->size();  // data size, not pointer size

        // TODO: now only support scalar, need to support array
        if (ir_gvar->is_init()) {  //! gvar init: .data
            MIRDataStorage::Storage data;
            auto val = dyn_cast<ir::Constant>(ir_gvar->scalar_value());
            if (type->is_int()) {
                if (type->is_i32()) {
                    data.push_back(static_cast<uint32_t>(val->i32()));
                    // how to handle i1?
                } else if (type->is_float32()) {
                    /* float to uint32_t, type cast, doesnt change the memory */
                    fu32.f = val->f32();
                    data.push_back(fu32.u);
                } else if (type->is_array()) {
                    // TODO: handle array init
                }
            } else if (type->is_float()) {
            }
            size_t align = 4;  // TODO: align
            auto mir_storage = std::make_unique<MIRDataStorage>(
                std::move(data), false, ir_gvar->name());
            auto mir_gobj = std::make_unique<MIRGlobalObject>(
                align, std::move(mir_storage), &mir_module);
            mir_module.global_objs().push_back(std::move(mir_gobj));

            // mir_module.global_objs().push_back(
            //     std::make_unique<MIRGlobalObject>(
            //         align,
            //         std::make_unique<MIRDataStorage>(std::move(data), false,
            //                                          ir_gvar->name()),
            //         &mir_module));

        } else {               //! gvar not init: .bss
            size_t align = 4;  // TODO: align
            auto mir_storage =
                std::make_unique<MIRZeroStorage>(size, ir_gvar->name());
            auto mir_gobj = std::make_unique<MIRGlobalObject>(
                align, std::move(mir_storage), &mir_module);
            mir_module.global_objs().push_back(std::move(mir_gobj));
        }
        gvar_map.emplace(ir_gvar, mir_module.global_objs().back().get());
    }

    // TODO: transformModuleBeforeCodeGen

    //! 3. codegen
    CodeGenContext codegen_ctx{target, target.get_datalayout(),
                               target.get_target_inst_info(),
                               target.get_target_frame_info(), MIRFlags{}};
    codegen_ctx.iselInfo = &target.get_target_isel_info();
    lowering_ctx._code_gen_ctx = &codegen_ctx;
    //! 4. lower all functions
    for (auto& ir_func : ir_module.funcs()) {  // for all funcs
        auto mir_func = func_map[ir_func];
        if (ir_func->blocks().empty())
            continue;

        /* 4.1: lower function body to generic MIR */
        create_mir_function(ir_func, mir_func, codegen_ctx, lowering_ctx);

        /* 4.2: instruction selection */
        ISelContext isel_ctx(codegen_ctx);
        isel_ctx.run_isel(mir_func);

        /* 4.3 register coalescing */

        /* 4.4 peephole optimization */

        /* 4.5 pre-RA legalization */

        /* 4.6 pre-RA scheduling, minimize register usage */

        /* 4.7 register allocation */
        codegen_ctx.registerInfo = new RISCVRegisterInfo();
        if (codegen_ctx.registerInfo) {
            // linearAllocator(*mir_func, codegen_ctx);
            // fastAllocator(*mir_func, codegen_ctx);
        }

        /* 4.8 stack allocation */
        if (codegen_ctx.registerInfo) {
            /* after sa, all stack objects are allocated with .offset */
            // allocateStackObjects(mir_func, codegen_ctx);
            codegen_ctx.flags.postSA = true;
        }

        /* 4.9 post-RA scheduling, minimize cycles */

        /* 4.10 post legalization */
        postLegalizeFunc(*mir_func, codegen_ctx);
        /* 4.11 verify */
    }
    /* module verify */
}

MIRFunction* create_mir_function(ir::Function* ir_func,
                                 MIRFunction* mir_func,
                                 CodeGenContext& codegen_ctx,
                                 LoweringContext& lowering_ctx) {
    // TODO: before lowering, ge some analysis pass result
    /* aligenment */
    /* range */
    /* dom */

    //! 1. map from ir to mir
    // std::unordered_map<ir::BasicBlock*, MIRBlock*> block_map;
    auto& block_map = lowering_ctx._block_map;
    std::unordered_map<ir::Value*, MIROperand*> storage_map;

    auto& target = codegen_ctx.target;
    auto& datalayout = target.get_datalayout();

    for (auto ir_block : ir_func->blocks()) {
        mir_func->blocks().push_back(std::make_unique<MIRBlock>(
            ir_block, mir_func,
            "label" + std::to_string(codegen_ctx.next_id_label())));
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

    //! 3. process alloca, new stack object for each alloca
    lowering_ctx.set_mir_block(block_map.at(ir_func->entry()));  // entry
    for (auto& ir_inst :
         ir_func->entry()->insts()) {  // note: all alloca in entry
        if (ir_inst->scid() != ir::Value::vALLOCA)
            continue;

        auto pointee_type =
            dyn_cast<ir::PointerType>(ir_inst->type())->base_type();
        uint32_t align = 4;  // TODO: align, need bind to ir object
        auto storage = mir_func->add_stack_obj(
            codegen_ctx.next_id(),  // id
            static_cast<uint32_t>(
                pointee_type->size()),  //! size, datalayout; if array??
            align,                      // align
            0,                          // offset
            StackObjectUsage::Local);
        storage_map.emplace(ir_inst, storage);
        // emit load stack object addr inst
        auto addr = lowering_ctx.new_vreg(lowering_ctx.get_ptr_type());
        auto ldsa_inst = new MIRInst{InstLoadStackObjectAddr};
        ldsa_inst->set_operand(0, addr);
        ldsa_inst->set_operand(1, storage);
        lowering_ctx.emit_inst(ldsa_inst);
        // map
        lowering_ctx.add_valmap(ir_inst, addr);
    }

    //! lowering blocks
    for (auto ir_block : ir_func->blocks()) {
        auto mir_block = block_map[ir_block];
        lowering_ctx.set_mir_block(mir_block);
        for (auto ir_inst : ir_block->insts()) {
            if (ir_inst->scid() == ir::Value::vALLOCA)
                continue;
            create_mir_inst(ir_inst, lowering_ctx);
        }
    }
    return mir_func;
}
void lower(ir::UnaryInst* ir_inst, LoweringContext& ctx);
void lower(ir::BinaryInst* ir_inst, LoweringContext& ctx);
void lower(ir::BranchInst* ir_inst, LoweringContext& ctx);
void lower(ir::LoadInst* ir_inst, LoweringContext& ctx);
void lower(ir::StoreInst* ir_inst, LoweringContext& ctx);
void lower(ir::GetElementPtrInst* ir_inst, LoweringContext& ctx);
void lower(ir::ICmpInst* ir_inst, LoweringContext& ctx);
void lower(ir::FCmpInst* ir_inst, LoweringContext& ctx);
void lower(ir::CallInst* ir_inst, LoweringContext& ctx);

//! return
void lower(ir::ReturnInst* ir_inst, LoweringContext& ctx);

//! branch
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
            std::cerr << "alloca not supported" << std::endl;
            break;
        case ir::Value::vLOAD:
            lower(dyn_cast<ir::LoadInst>(ir_inst), ctx);
            break;
        case ir::Value::vSTORE:
            lower(dyn_cast<ir::StoreInst>(ir_inst), ctx);
            break;
        case ir::Value::vGETELEMENTPTR:
            lower(dyn_cast<ir::GetElementPtrInst>(ir_inst), ctx);
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
        default:
            assert(false && "not supported inst");
            break;
    }
    return nullptr;
}
void lower(ir::UnaryInst* ir_inst, LoweringContext& ctx) {
    // TODO: Need Test
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
    auto ret = ctx.new_vreg(ir_inst->type());
    auto inst = new MIRInst(InstICmp);
    inst->set_operand(0, ret);
    inst->set_operand(1, ctx.map2operand(ir_inst->operand(0)));
    inst->set_operand(2, ctx.map2operand(ir_inst->operand(1)));
    // TODO: set condition code: eq, ne, sgt, sge, slt, sle
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
    auto ret = ctx.new_vreg(ir_inst->type());
    auto inst = new MIRInst(InstFCmp);
    inst->set_operand(0, ret);
    inst->set_operand(1, ctx.map2operand(ir_inst->operand(0)));
    inst->set_operand(2, ctx.map2operand(ir_inst->operand(1)));
    // TODO: set condition code: oeq, one, ogt, oge, olt, ole
    ctx.emit_inst(inst);
    ctx.add_valmap(ir_inst, ret);
}

void lower(ir::CallInst* ir_inst, LoweringContext& ctx) {
    // TODO: Need Test

    ctx._target.get_target_frame_info().emit_call(ir_inst, ctx);
}

void lower(ir::GetElementPtrInst* ir_inst, LoweringContext& ctx) {
    // TODO: implement getelementptr
    std::cerr << "getelementptr not supported" << std::endl;
}

//! BinaryInst
void lower(ir::BinaryInst* ir_inst, LoweringContext& ctx) {
    // MIRGenericInst gc_instid;
    auto gc_instid = [scid = ir_inst->scid()] {
        switch (scid) {
            case ir::Value::vADD:
                return InstAdd;
            case ir::Value::vFADD:
                return InstFAdd;
            case ir::Value::vSUB:
                return InstSub;
            case ir::Value::vFSUB:
                return InstFSub;
            case ir::Value::vMUL:
                return InstMul;
            case ir::Value::vFMUL:
                return InstFMul;
            case ir::Value::vUDIV:
                return InstUDiv;
            case ir::Value::vSDIV:
                return InstSDiv;
            case ir::Value::vFDIV:
                return InstFDiv;
            case ir::Value::vUREM:
                return InstURem;
            case ir::Value::vSREM:
                return InstSRem;
            default:
                assert(false && "not supported binary inst");
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

//! BranchInst
void emit_branch(ir::BasicBlock* srcblock,
                 ir::BasicBlock* dstblock,
                 LoweringContext& lctx);

/**
 *
 */
void lower(ir::BranchInst* ir_inst, LoweringContext& ctx) {
    auto src_block = ir_inst->parent();

    if (ir_inst->is_cond()) {
        // TODO: conditional branch
        int a = 5;
        int b = 10;
        std::cout << "conditional branch" << std::endl;
    } else {  // unconditional branch
        auto dst_block = ir_inst->dest();
        emit_branch(src_block, dst_block, ctx);
    }
}
/**
 * IR:
 *  br dest
 *  br cond, iftrue, iffalse
 * MIRGeneric:
 *  Jump {target}
 *  Branch {cond}, {target}, {prob}
 */
void emit_branch(ir::BasicBlock* srcblock,
                 ir::BasicBlock* dstblock,
                 LoweringContext& lctx) {
    auto dst_mblock = lctx.map2block(dstblock);
    auto src_mblock = lctx.map2block(srcblock);
    auto operand = MIROperand::as_reloc(dst_mblock);
    auto inst = new MIRInst(InstJump);
    inst->set_operand(0, operand);
    lctx.emit_inst(inst);
}
//! LoadInst and StoreInst
void lower(ir::LoadInst* ir_inst, LoweringContext& ctx) {
    // ir: %13 = load i32, i32* %12
    //! ret = load ptr
    // load ret, ptr
    auto ret = ctx.new_vreg(ir_inst->type());
    auto ptr = ctx.map2operand(ir_inst->operand(0));
    assert(ret != nullptr && ptr != nullptr);
    uint32_t align = 4;
    auto inst = new MIRInst(InstLoad);
    inst->set_operand(0, ret);
    inst->set_operand(1, ptr);
    inst->set_operand(2, MIROperand::as_imm(align, OperandType::Special));

    ctx.emit_inst(inst);
    ctx.add_valmap(ir_inst, ret);
}

void lower(ir::StoreInst* ir_inst, LoweringContext& ctx) {
    // auto& ir_store = dyn_cast<ir::StoreInst>(ir_inst);
    // ir: store type val, type* ptr
    //! store addr, val
    // align = 4
    auto inst = new MIRInst(InstStore);
    inst->set_operand(0, ctx.map2operand(ir_inst->ptr()));
    inst->set_operand(1, ctx.map2operand(ir_inst->value()));
    uint32_t align = 4;
    inst->set_operand(2, MIROperand::as_imm(align, OperandType::Special));

    ctx.emit_inst(inst);
}
//! ReturnInst
void lower(ir::ReturnInst* ir_inst, LoweringContext& ctx) {
    ctx._target.get_target_frame_info().emit_return(ir_inst, ctx);
}

}  // namespace mir
