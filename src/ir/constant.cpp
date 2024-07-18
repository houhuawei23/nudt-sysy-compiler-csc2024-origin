#include "ir/constant.hpp"
#include "ir/utils_ir.hpp"

namespace ir {

//! Constant
//* Instantiation for static data attribute
std::map<std::string, Constant*> Constant::cache;

void Constant::print(std::ostream& os) const {
    if (type()->isInt32()) {
        os << i32();
    } else if (type()->isFloatPoint()) {
        os << name();  // 0x...
    } else if (type()->isUnder()) {
        os << "undef";
    } else {
        assert(false);
    }
}
}  // namespace ir