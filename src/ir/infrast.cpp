#include "include/infrast.hpp"
#include "include/utils.hpp"

namespace ir {

std::map<std::string, Constant*> Constant::cache;

void BasicBlock::print(std::ostream &os) const {
    // os <<
    // print all instructions
    std::string n = name();
    if (n.size() > 0) {
        n = n.substr(1);
    }
    os << n << ":" << "     " << "; block" << std::endl;
    for (auto &inst : _insts) {
        os << "    " << *inst << std::endl;
    }
}

void Constant::print(std::ostream &os) const {
    if (type()->is_i32()) {
        os << i32();
    } else if (type()->is_float()) {
        os << f64();
    } else {
        assert(false);
    }
}

void Argument::print(std::ostream &os) const {
    os << *type() << " " << name();
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


} // namespace ir