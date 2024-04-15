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
    auto& mir_module = lowering_ctx._mir_module; // return ref, using ref to use
    auto& functions = mir_module.functions();
    auto& global_objs = mir_module.global_objs();
    
    for (auto func : ir_module.funcs()) {
        functions.push_back(std::make_unique<MIRFunction>(func->name(), &mir_module));
        //! error, why?
        // mir_module.functions().push_back(std::make_unique<MIRFunction>(func->name(), mir_module));
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
            auto mir_storage = std::make_unique<MIRDataStorage>(std::move(data), false);
            auto mir_gobj = std::make_unique<MIRGlobalObject>(align, std::move(mir_storage), &mir_module);
            mir_module.global_objs().push_back(std::move(mir_gobj));
        } else {  //! gvar not init: .bss

            size_t align = 4;  // TODO: align
            auto mir_storage = std::make_unique<MIRZeroStorage>(size, ir_gvar->name());
            auto mir_gobj = std::make_unique<MIRGlobalObject>(align, std::move(mir_storage), &mir_module);
            mir_module.global_objs().push_back(std::move(mir_gobj));
        }
        lowering_ctx.gvar_map.emplace(ir_gvar, mir_module.global_objs().back().get());
    }

    // TODO: transformModuleBeforeCodeGen

    //! codegen
    auto& target = lowering_ctx._target;  // Target*
    CodeGenContext codegen_ctx{target, target.get_datalayout(),
                               target.get_inst_info(), target.get_frame_info(),
                               MIRFlags{}};
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
        // ISelContext isel_ctx(&codegen_ctx);
        // isel_ctx.run_isel(mir_func);

        // register coalescing
        // peephole optimization
        // register allocation
        // stack allocation
    }
}

MIRFunction* create_mir_function(ir::Function* ir_func,
                                 MIRFunction* mir_func,
                                 LoweringContext& lowering_ctx) {
    auto codegen_ctx = lowering_ctx._code_gen_ctx;  // *

    std::unordered_map<ir::BasicBlock*, MIRBlock*> block_map;
    std::unordered_map<ir::Value*, MIROperand> value_map;
    std::unordered_map<ir::Value*, MIROperand> storage_map;
    // map all blocks
    for (auto ir_block : ir_func->blocks()) {
        auto mir_block = new MIRBlock(ir_block, mir_func);
        mir_func->blocks().push_back(mir_block);
        block_map[ir_block] = mir_block;
    }

    // args
    for (auto ir_arg : ir_func->args()) {
        // assign vreg to arg
    }

    //! emitPrologue for function
    {
        for (auto arg : ir_func->args()) {
            auto vreg = lowering_ctx.new_vreg(arg->type());
            lowering_ctx.add_valmap(arg, vreg);
            mir_func->args().push_back(vreg);
        }
        lowering_ctx.set_mir_block(block_map.at(ir_func->entry()));
        codegen_ctx->frameInfo.emit_prologue(mir_func, lowering_ctx);
    }

    //! process alloca
    lowering_ctx.set_mir_block(block_map.at(ir_func->entry()));
    for (auto& inst : ir_func->entry()->insts()) {
        // all alloca in entry
        auto type = dyn_cast<ir::PointerType>(inst->type());
        // auto storage = mir_func->add_stack_obj(ctx.next_id(), )
    }

    // blocks
    for (auto ir_block : ir_func->blocks()) {
        auto mir_block = block_map[ir_block];
        // set current block
        for (auto ir_inst : ir_block->insts()) {
            create_mir_inst(ir_inst, lowering_ctx);
        }
    }
    return mir_func;
}

MIRInst* create_mir_inst(ir::Instruction* ir_inst, LoweringContext& ctx) {
    switch (ir_inst->scid()) {
        case ir::Value::vADD: {
            auto ret = ctx.new_vreg(ir_inst->type());
            auto inst = new MIRInst(InstAdd);
            inst->set_operand(0, ret);
            inst->set_operand(1, ctx.map2operand(ir_inst->operand(0)));
            inst->set_operand(2, ctx.map2operand(ir_inst->operand(1)));
            ctx.emit_inst(inst);
            break;
        }

        case ir::Value::vALLOCA: {
            // emit_inst()
            break;
        }
        case ir::Value::vLOAD: {
            // ir: %13 = load i32, i32* %12
            auto ret = ctx.new_vreg(ir_inst->type());
            auto ptr = ctx.map2operand(ir_inst->operand(0));
            auto align = 4;
            auto inst = new MIRInst(InstLoad);
            inst->set_operand(0, ret);
            inst->set_operand(1, ptr);

            ctx.emit_inst(inst);
            //   .set_operand(2, MIROperand::as_imm(
            //                       align, OperandType::Special))
            break;
        }
        case ir::Value::vSTORE: {
            // auto& ir_store = dyn_cast<ir::StoreInst>(ir_inst);
            // ir: store type val, type* ptr
            // align = 4
            auto inst = new MIRInst(InstStore);
            inst->set_operand(0, ctx.map2operand(ir_inst->operand(0)));
            inst->set_operand(1, ctx.map2operand(ir_inst->operand(1)));

            ctx.emit_inst(inst);
            break;
        }
        case ir::Value::vRETURN: {
            auto inst = new MIRInst(InstRet);
        }
        default:
            break;
    }
    return nullptr;
}

void lower(ir::BranchInst* ir_inst, LoweringContext& ctx) {
    if (!ir_inst->is_cond()) {
        // unconditional branch
        // emit_branch();
    }
}

}  // namespace mir
