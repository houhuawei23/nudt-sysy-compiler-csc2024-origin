#include "mir/mir.hpp"
#include "ir/ir.hpp"
namespace mir {
MIRModule::MIRModule(ir::Module* ir_module) {
    _ir_module = ir_module;
    for (auto func : ir_module->funcs()) {
        auto mir_func = new MIRFunction(func, this);
    }
    // for (auto gv : ir_module->gvalues()) {
    //     auto mir_gv = new MIRGlobalValue(gv, this);
    // }
}

MIRFunction::MIRFunction(ir::Function* ir_func, MIRModule* parent){
    _parent = parent;
    _ir_func = ir_func;
}

MIRBlock::MIRBlock(ir::BasicBlock* ir_block, MIRFunction* parent){
    _parent = parent;
    _ir_block = ir_block;
}


}  // namespace mir
