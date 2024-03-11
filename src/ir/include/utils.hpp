#include "type.hpp"
#include "value.hpp"
#include <iostream>

namespace ir {

// for simple use <<
inline std::ostream &operator<<(std::ostream &os, const Type &type) {
    type.print(os);
    return os;
}

inline std::ostream &operator<<(std::ostream &os, const Value &value) {
    value.print(os);
    return os;
}
} // namespace ir
