#include "mir/mir.hpp"
#include "mir/lowering.hpp"
#include "mir/target.hpp"
#include "mir/iselinfo.hpp"
namespace mir {
//! declare
void create_mir_module(ir::Module& ir_module, LoweringContext& ctx);

MIRFunction* create_mir_function(ir::Function* ir_func,
                                 MIRFunction* mir_func,
                                 LoweringContext& ctx);

MIRInst* create_mir_inst(ir::Instruction* ir_inst, LoweringContext& ctx);

//! implementation

std::unique_ptr<MIRModule> create_mir_module(ir::Module& ir_module,
                                             Target& target) {
    // new MIRModule
    auto mir_module_uptr = std::make_unique<MIRModule>(&ir_module, target);
    LoweringContext ctx = LoweringContext(*mir_module_uptr, target);
    create_mir_module(ir_module, ctx);
    return mir_module_uptr;
}

void create_mir_module(ir::Module& ir_module, LoweringContext& lowering_ctx) {
    // return ref, using ref to use
    auto& mir_module = lowering_ctx._mir_module;
    auto& functions = mir_module.functions();
    auto& global_objs = mir_module.global_objs();

    for (auto func : ir_module.funcs()) {
        functions.push_back(
            std::make_unique<MIRFunction>(func->name(), &mir_module));
        // auto& mir_func = functions.back();
        lowering_ctx.func_map.emplace(func, functions.back().get());
    }
    //! for all global variables, create MIRGlobalObject
    for (auto ir_gval : ir_module.gvalues()) {
        auto ir_gvar = dyn_cast<ir::GlobalVariable>(ir_gval);
        auto type = dyn_cast<ir::PointerType>(ir_gvar->type())->base_type();
        size_t size = ir_gvar->type()->size();

        if (ir_gvar->is_init()) {  //! gvar init: .data
            // TODO: now only support scalar, need to support array
            MIRDataStorage::Storage data;
            auto val = dyn_cast<ir::Constant>(ir_gvar->scalar_value());
            if (type->is_int()) {
                if (type->is_i32()) {
                    data.push_back(static_cast<uint32_t>(val->i32()));
                }
            } else if (type->is_float()) {
            }
            size_t align = 4;  // TODO: align
            auto mir_storage =
                std::make_unique<MIRDataStorage>(std::move(data), false);
            auto mir_gobj = std::make_unique<MIRGlobalObject>(
                align, std::move(mir_storage), &mir_module);
            mir_module.global_objs().push_back(std::move(mir_gobj));
        } else {  //! gvar not init: .bss

            size_t align = 4;  // TODO: align
            auto mir_storage =
                std::make_unique<MIRZeroStorage>(size, ir_gvar->name());
            auto mir_gobj = std::make_unique<MIRGlobalObject>(
                align, std::move(mir_storage), &mir_module);
            mir_module.global_objs().push_back(std::move(mir_gobj));
        }
        lowering_ctx.gvar_map.emplace(ir_gvar,
                                      mir_module.global_objs().back().get());
    }

    // TODO: transformModuleBeforeCodeGen

    //! codegen
    auto& target = lowering_ctx._target;  // Target*
    CodeGenContext codegen_ctx{target, target.get_datalayout(),
                               target.get_target_inst_info(),
                               target.get_target_frame_info(), MIRFlags{}};
    // target.get_target_isel_info(),  //! segmentation fault, not implemented
    // target.get_register_info(),
    codegen_ctx.iselInfo = &target.get_target_isel_info();
    // codegen_ctx.registerInfo = &target.get_register_info();
    lowering_ctx.set_code_gen_ctx(&codegen_ctx);
    //! lower all functions
    for (auto& ir_func : ir_module.funcs()) {  // for all funcs
        auto mir_func = lowering_ctx.func_map[ir_func];
        if (ir_func->blocks().empty()) {
            continue;
        }
        // 1: lower function body to generic MIR
        create_mir_function(ir_func, mir_func, lowering_ctx);

        // 2: inst selection
        ISelContext isel_ctx(codegen_ctx);
        isel_ctx.run_isel(mir_func);

        /* register coalescing */

        /* peephole optimization */

        /* pre-RA legalization */

        /* pre-RA scheduling, minimize register usage */

        /* register allocation */
        if (codegen_ctx.registerInfo) {
            // allocate_registers(mir_func, codegen_ctx);
        }

        /* stack allocation */

        /* post-RA scheduling, minimize cycles */

        /* post legalization */

        /* verify */
    }
    /* module verify */
}

MIRFunction* create_mir_function(ir::Function* ir_func,
                                 MIRFunction* mir_func,
                                 LoweringContext& lowering_ctx) {
    auto codegen_ctx = lowering_ctx._code_gen_ctx;  // *

    // TODO: before lowering, ge some analysis pass result
    /* aligenment */
    /* range */
    /* dom */

    // map from ir to mir
    // std::unordered_map<ir::BasicBlock*, MIRBlock*> block_map;
    auto& block_map = lowering_ctx._block_map;  // return ref, using ref to use

    std::unordered_map<ir::Value*, MIROperand*> storage_map;

    auto& target = lowering_ctx._target;         // Target&
    auto& datalayout = target.get_datalayout();  // DataLayout&

    // map all blocks
    for (auto ir_block : ir_func->blocks()) {  // dom.blocks()?
        mir_func->blocks().push_back(
            std::make_unique<MIRBlock>(ir_block, mir_func));
        block_map.emplace(ir_block, mir_func->blocks().back().get());
    }

    //! emitPrologue for function
    {  // args
        for (auto ir_arg : ir_func->args()) {
            // assign vreg to arg
            auto vreg = lowering_ctx.new_vreg(ir_arg->type());
            lowering_ctx.add_valmap(ir_arg, vreg);
            mir_func->args().push_back(vreg);
        }
        lowering_ctx.set_mir_block(block_map.at(ir_func->entry()));  // entry
        // TODO: implement riscv frameinfo.emit_prologue()
        codegen_ctx->frameInfo.emit_prologue(mir_func, lowering_ctx);
    }

    //! process alloca, new stack object for each alloca
    lowering_ctx.set_mir_block(block_map.at(ir_func->entry()));  // entry
    for (auto& ir_inst : ir_func->entry()->insts()) {
        // all alloca in entry
        if (ir_inst->scid() != ir::Value::vALLOCA)
            continue;
        // else: alloca inst
        auto pointee_type =
            dyn_cast<ir::PointerType>(ir_inst->type())->base_type();
        uint32_t align = 4;  // TODO: align, need bind to ir object
        auto storage = mir_func->add_stack_obj(
            lowering_ctx.next_id(),  // id
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

    // lowering blocks
    for (auto ir_block : ir_func->blocks()) {
        auto mir_block = block_map[ir_block];
        // set current block
        lowering_ctx.set_mir_block(mir_block);
        for (auto ir_inst : ir_block->insts()) {
            // lowering inst
            if (ir_inst->scid() == ir::Value::vALLOCA)
                continue;  // jump alloca
            create_mir_inst(ir_inst, lowering_ctx);
        }
    }
    return mir_func;
}

void lower(ir::BinaryInst* ir_inst, LoweringContext& ctx);

void lower(ir::BranchInst* ir_inst, LoweringContext& ctx);

void lower(ir::LoadInst* ir_inst, LoweringContext& ctx);

void lower(ir::StoreInst* ir_inst, LoweringContext& ctx);

//! return
void lower(ir::ReturnInst* ir_inst, LoweringContext& ctx);

//! branch
void lower(ir::BranchInst* ir_inst, LoweringContext& ctx);

MIRInst* create_mir_inst(ir::Instruction* ir_inst, LoweringContext& ctx) {
    // auto scid = ir_inst->scid();
    // if (scid > ir::Value::vBINARY_BEGIN and scid < ir::Value::vBINARY_END) {
    //     lower(dyn_cast<ir::BinaryInst>(ir_inst), ctx);
    // } else if (scid > ir::Value::vUNARY_BEGIN and scid <
    // ir::Value::vUNARY_END) {
    //     lower(dyn_cast<ir::UnaryInst>(ir_inst), ctx);
    // } else if (scid > ir::Value::vICMP_BEGIN and scid < ir::Value::vICMP_END)
    // {
    //     lower(dyn_cast<ir::ICmpInst>(ir_inst), ctx);
    // } else if (scid > ir::Value::vFCMP_BEGIN and scid < ir::Value::vFCMP_END)
    // {
    //     lower(dyn_cast<ir::FCmpInst>(ir_inst), ctx);
    // }

    switch (ir_inst->scid()) {
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
        default:
            assert(false && "not supported inst");
            break;
    }
    return nullptr;
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
            // case ir::Value::vFREM:
            //     return InstFRem;
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

void lower(ir::BranchInst* ir_inst, LoweringContext& ctx) {
    auto src_block = ir_inst->parent();

    if (ir_inst->is_cond()) {
        // TODO: conditional branch
    } else {
        // unconditional branch
        auto dst_block = ir_inst->dest();
        emit_branch(src_block, dst_block, ctx);
    }
}

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
    auto align = 4;
    auto inst = new MIRInst(InstLoad);
    inst->set_operand(0, ret);
    inst->set_operand(1, ptr);

    ctx.emit_inst(inst);
    ctx.add_valmap(ir_inst, ret);
}

void lower(ir::StoreInst* ir_inst, LoweringContext& ctx) {
    // auto& ir_store = dyn_cast<ir::StoreInst>(ir_inst);
    // ir: store type val, type* ptr
    //! store val, ptr
    // align = 4
    auto inst = new MIRInst(InstStore);
    inst->set_operand(0, ctx.map2operand(ir_inst->value()));
    inst->set_operand(1, ctx.map2operand(ir_inst->ptr()));

    ctx.emit_inst(inst);
}
//! ReturnInst
void lower(ir::ReturnInst* ir_inst, LoweringContext& ctx) {
    ctx._target.get_target_frame_info().emit_return(ir_inst, ctx);
}

}  // namespace mir
