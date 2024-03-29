#include "mir/mir.hpp"
#include "ir/ir.hpp"
namespace mir {
MIRModule::MIRModule(ir::Module* ir_module) {
    _ir_module = ir_module;
    for (auto func : ir_module->funcs()) {
        auto mir_func = new MIRFunction(func, this);
        _functions.push_back(mir_func);
    }
    for (auto gv : ir_module->gvalues()) {
        auto mir_gv = new MIRGlobalObject(gv, this);
        _global_objs.push_back(mir_gv);
    }
}

MIRFunction::MIRFunction(ir::Function* ir_func, MIRModule* parent){
    _parent = parent;
    _ir_func = ir_func;
    _name = ir_func->name();

    MIRBlock* entry_mirblock = new MIRBlock(ir_func->entry(), this);
    _blocks.emplace_back(entry_mirblock);


    for (auto bb : ir_func->blocks()) {
        auto mir_block = new MIRBlock(bb, this);
        mir_block->inst_sel(bb);
        _blocks.emplace_back(mir_block);
        

    }


}

MIRBlock::MIRBlock(ir::BasicBlock* ir_block, MIRFunction* parent){
    _parent = parent;
    _ir_block = ir_block;
}



void MIRBlock::inst_sel(ir::BasicBlock* ir_bb){
    
    for (auto inst : ir_bb->insts()) {
        
    }
}
 MIRGlobalObject::MIRGlobalObject(ir::Value* ir_gv, MIRModule* parent) {

 };


}  // namespace mir
