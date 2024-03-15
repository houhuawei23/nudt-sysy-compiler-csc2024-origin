#include "include/infrast.hpp"
#include "include/utils.hpp"

namespace ir {
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
    if (type()->is_int()) {
        os << i();
    } else if (type()->is_float()) {
        os << f();
    } else {
        // assert
    }
}
} // namespace ir