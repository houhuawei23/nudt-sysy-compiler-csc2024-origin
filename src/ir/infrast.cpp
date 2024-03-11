#include "include/infrast.hpp"
#include "include/utils.hpp"

namespace ir {
void BasicBlock::print(std::ostream &os) const {
    // os <<
    // print all instructions
    for (auto &inst : _instructions) {
        os << "    " << *inst << std::endl;
    }
}

void Constant::print(std::ostream &os) const {
    if (get_type()->is_int()) {
        os << get_int();
    } else if (get_type()->is_float()) {
        os << get_float();
    } else {
        // assert
    }
}
} // namespace ir