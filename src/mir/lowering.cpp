#include "mir/mir.hpp"
#include "mir/lowering.hpp"

namespace mir {
//! declare
void create_mir_module(ir::Module* ir_module, LoweringContext& ctx);

MIRFunction* create_mir_function(ir::Function* ir_func,
                                 MIRFunction* mir_func,
                                 LoweringContext& ctx);

MIRInst* create_mir_inst(ir::Instruction* ir_inst, LoweringContext& ctx);

//! implementation

MIRModule* create_mir_module(ir::Module* ir_module) {
    MIRModule* mir_module = new MIRModule(ir_module);
    LoweringContext ctx = LoweringContext(mir_module);
    create_mir_module(ir_module, ctx);
    return mir_module;
}

void create_mir_module(ir::Module* ir_module, LoweringContext& ctx) {
    auto& mir_module = ctx._mir_module;
    for (auto func : ir_module->funcs()) {
        // auto mir_func = new MIRFunction(func->name(), mir_module);
        // only add function declar, or recursively generate function body?
        auto mir_func = mir_module->add_func(func->name());
        ctx.func_map[func] = mir_func;  // update map
    }

    for (auto ir_gval : ir_module->gvalues()) {
        auto ir_gvar = dynamic_cast<ir::GlobalVariable*>(ir_gval);
        auto type = dyn_cast<ir::PointerType>(ir_gvar->type())->base_type();
        size_t size = ir_gvar->type()->size();

        if (ir_gvar->is_init()) {  // gvar init: .data
            MIRDataStorage::Storage data;
            auto val = dynamic_cast<ir::Constant*>(ir_gvar->init(0));
            if (type->is_int()) {
                if (type->is_i32()) {
                    data.push_back(static_cast<uint32_t>(val->i32()));
                }
            } else if (type->is_float()) {
            }
            size_t align = 4;  // TODO: align
            auto mir_storage = new MIRDataStorage(std::move(data), false);
            auto mir_gobj = new MIRGlobalObject(4, mir_storage, mir_module);
            mir_module->add_gobj(mir_gobj);
        } else {  // gvar not init: .bss

            size_t align = 4;  // TODO: align
            auto mir_storage = new MIRZeroStorage(size, ir_gvar->name());
            auto mir_gobj = new MIRGlobalObject(align, mir_storage, mir_module);
            mir_module->add_gobj(mir_gobj);
        }
        // gvar_map.emplace(ir_gvar, mir_module->global_objs().back());
        ctx.gvar_map[ir_gvar] = mir_module->global_objs().back();  // update map
    }

    for (auto ir_func : ir_module->funcs()) {  // for all funcs
        auto mir_func = ctx.func_map[ir_func];
        if (ir_func->blocks().empty()) {
            continue;
        }
        create_mir_function(ir_func, mir_func, ctx);
    }
}

MIRFunction* create_mir_function(ir::Function* ir_func,
                                 MIRFunction* mir_func,
                                 LoweringContext& ctx) {
    std::unordered_map<ir::BasicBlock*, MIRBlock*> block_map;
    std::unordered_map<ir::Value*, MIROperand> value_map;
    std::unordered_map<ir::Value*, MIROperand> storage_map;

    for (auto ir_block : ir_func->blocks()) {
        auto mir_block = new MIRBlock(ir_block, mir_func);
        mir_func->blocks().push_back(mir_block);
        block_map[ir_block] = mir_block;
    }

    // args
    for (auto ir_arg : ir_func->args()) {
        // assign vreg to arg
    }

    //! process alloca
    ctx.set_mir_block(block_map.at(ir_func->entry()));
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
            create_mir_inst(ir_inst, ctx);
        }
    }
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
