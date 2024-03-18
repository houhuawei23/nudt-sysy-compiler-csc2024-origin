#pragma onece
#include "include/global.hpp"

namespace ir {
void GlobalVariable::print(std::ostream& os) const {
    os << "@" << name() << " = ";
    if (is_decl_const()) {
        os << "constant ";
    } else {
        os << "global ";
    }
    os << *dyn_cast<PointerType>(type())->base_type() << " ";
    if (is_init()) {
        os << *init_value();
    }
    os << "\n";
}
}  // namespace ir