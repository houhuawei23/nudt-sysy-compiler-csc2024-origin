#pragma once

namespace sysy {
#define any_cast_Type std::any_cast<ir::Type *>
#define any_cast_Value std::any_cast<ir::Value *>

#include <any>
#include <typeinfo>
#include <iostream>

template<typename T>
T* safe_any_cast(std::any any_value) {
    if (any_value.type() == typeid(T*)) {
        return std::any_cast<T*>(any_value);
    } else {
        // Handle the mismatched type gracefully
        // std::cerr << "Error: Type mismatch during safe_any_cast." << std::endl;
        return nullptr;
    }
}
} // namespace sysy
