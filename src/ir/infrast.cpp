#include "ir/infrast.hpp"
#include "ir/utils_ir.hpp"
#include "ir/function.hpp"

namespace ir {


//! Constant
//* Instantiation for static data attribute
// std::map<std::string, Constant*> Constant::cache;


// void Constant::print(std::ostream &os) {
//     if (type()->is_i32()) {
//         os << i32();
//     } else if (type()->is_float()) {
//         os << f64();
//     } else {
//         assert(false);
//     }
// }

void Argument::print(std::ostream &os){
    os << *type() << " " << name();
}

void BasicBlock::print(std::ostream &os) {
    // print all instructions

    os << name() << ":" << "     " << "; block" << std::endl;
    for (auto &inst : _insts) {
        os << "    " << *inst << std::endl;
    }
}

void BasicBlock::emplace_back_inst(Instruction* i) {
    if(not _is_terminal)
        _insts.emplace_back(i); 
    _is_terminal=i->is_terminator();
}
void BasicBlock::emplace_inst(inst_iterator pos, Instruction* i) {
    if(not _is_terminal)
        _insts.emplace(pos, i);
    _is_terminal=i->is_terminator();
}

void Instruction::setvarname(){
    auto cur_func = _parent->parent();
    _name= "%" + std::to_string(cur_func->getvarcnt());
}

} // namespace ir