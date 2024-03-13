#pragma once

namespace sysy {
#define any_cast_Type std::any_cast<ir::Type *>
#define any_cast_Value std::any_cast<ir::Value *>


//! be careful
template <typename To, typename From>
[[nodiscard]] inline decltype(auto) dyn_cast(From *Val) {
  return dynamic_cast<To *>(Val);
}
} // namespace sysy
