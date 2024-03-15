#pragma once

#include <iostream>
#include <sstream>
#include <iomanip>
#include "type.hpp"
#include "value.hpp"

namespace ir {

// for simple use <<
inline std::ostream& operator<<(std::ostream& os, const Type& type) {
    type.print(os);
    return os;
}

inline std::ostream& operator<<(std::ostream& os, const Value& value) {
    value.print(os);
    return os;
}
// type check, eg:
// ir::isa<ir::Function>(func)
template <typename T>
inline std::enable_if_t<std::is_base_of_v<Value, T>, bool> isa(
    const Value* value) {
    return T::classof(value);
}

//! be careful
template <typename To, typename From>
[[nodiscard]] inline decltype(auto) dyn_cast(From *Val) {
  return dynamic_cast<To *>(Val);
}
// get machine code
inline std::string getMC(float f){
    double d=f;
    unsigned int mrf=*reinterpret_cast<unsigned int*>(&f);
    std::stringstream ss;
    ss << std::hex << std::uppercase << std::setfill('0') << std::setw(16) << mrf;
    std::string res="0x"+ss.str();
    return res;
}

}  // namespace ir
