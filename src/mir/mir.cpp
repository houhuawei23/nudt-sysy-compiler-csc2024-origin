#include "mir/mir.hpp"
#include "ir/ir.hpp"

namespace mir {

/*
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
void MIRModule::print(std::ostream& os) {
    for (auto gv : _global_objs) {
        gv->print(os);
        os << "\n";
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
*/

/*
 * @brief Global Value Assembly Code
 *      1. Variable (GlobalVariable)
 *          1.1 Scalar
 *          1.2 Array (TODO)
 *      2. Constant (TODO)
 *          2.1 Scalar (Constant)
 *          2.2 Array (GlobalVariable)
 * @example
 *      a:
 *          .word   2
 */
/*
void MIRGlobalObject::print(std::ostream& os) {
    os << _ir_global->name() << ":\n";

    if (ir::isa<ir::GlobalVariable>(_ir_global)) {
        auto global = dyn_cast<ir::GlobalVariable>(_ir_global);
        auto scalar = global->scalar_value();
        os << "  .word" << " ";
        if (ir::isa<ir::Constant>(scalar)) {
            auto cscalar = dyn_cast<ir::Constant>(scalar);
            if (cscalar->is_i32()) os << cscalar->i32();
            else if (cscalar->is_float()) os << ir::getMC(cscalar->f32());
            else assert(false && "invalid type");   
        }
    } else {
        assert(false && "the variable is not global scope");
    }
}
*/
}  // namespace mir
