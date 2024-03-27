#include "ir/constant.hpp"
#include "ir/utils_ir.hpp"

namespace ir {

//! Constant
//* Instantiation for static data attribute
std::map<std::string, Constant*> Constant::cache;

void Constant::print(std::ostream& os) {
    if (type()->is_i32()) {
        os << i32();
    } else if (type()->is_float()) {
        os << f64();
    } else {
        assert(false);
    }
}
}  // namespace ir