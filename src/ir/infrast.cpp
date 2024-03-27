#include "ir/infrast.hpp"
#include "ir/function.hpp"
#include "ir/utils_ir.hpp"

namespace ir {

void Argument::print(std::ostream& os) {
    os << *type() << " " << name();
}

void BasicBlock::print(std::ostream& os) {
    // print all instructions

    os << name() << ":" << std::endl;

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
    if (not _is_terminal)
        _insts.emplace(pos, i);
    _is_terminal = i->is_terminator();
}

void Instruction::setvarname() {
    auto cur_func = _parent->parent();
    _name = "%" + std::to_string(cur_func->getvarcnt());
}

}  // namespace ir