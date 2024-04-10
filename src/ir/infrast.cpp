#include "ir/infrast.hpp"
#include "ir/function.hpp"
#include "ir/utils_ir.hpp"
#include "ir/instructions.hpp"

namespace ir {

void Argument::print(std::ostream& os) {
    os << *type() << " " << name();
}

void BasicBlock::print(std::ostream& os) {
    // print all instructions

    os << name() << ":" ;
    if(!_comment.empty()) {
        os << " ; " << _comment << std::endl;
    } else {
        os << std::endl;
    }
    if (pre_num()) {
        os << "    ; "
           << "pres: ";
        for (auto it = pre_blocks().begin(); it != pre_blocks().end(); it++) {
            os << (*it)->name();
            if (std::next(it) != pre_blocks().end()) {
                os << ", ";
            }
        }
        os << std::endl;
    }
    if (next_num()) {
        os << "    ; "
           << "nexts: ";
        for (auto it = next_blocks().begin(); it != next_blocks().end(); it++) {
            os << (*it)->name();
            if (std::next(it) != next_blocks().end()) {
                os << ", ";
            }
        }
        os << std::endl;
    }

    for (auto& inst : _insts) {
        os << "    " << *inst << std::endl;
    }
}

void BasicBlock::emplace_back_inst(Instruction* i) {
    if (not _is_terminal)
        _insts.emplace_back(i);
    _is_terminal = i->is_terminator();
}
void BasicBlock::emplace_inst(inst_iterator pos, Instruction* i) {
    //Warning: didn't check _is_terminal
    _insts.emplace(pos, i);
    _is_terminal = i->is_terminator();
}

void BasicBlock::delete_inst(Instruction* inst){
    for(auto op:inst->operands()){
        auto val=op->value();
        if(val){
            val->del_use(op);
        }
    }
    delete inst;
}

void Instruction::setvarname() {
    auto cur_func = _parent->parent();
    _name = "%" + std::to_string(cur_func->getvarcnt());
}

// void BasicBlock::delete_inst(Instruction* inst){
//     for(auto puse:inst->uses()){
//         puse->user()->del_use(puse);
//     }
//     _insts.remove(inst);
//     delete inst;
// }

void BasicBlock::emplace_first_inst(Instruction* inst){
    //Warning: didn't check _is_terminal
    auto pos=_insts.begin();
    _insts.emplace(pos,inst);
    _is_terminal = inst->is_terminator();
}

}  // namespace ir