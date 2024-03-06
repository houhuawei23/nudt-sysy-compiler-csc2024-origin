#include "include/module.hpp"
#include <cassert>

namespace ir {
Function *Module::get_function(const_str_ref name) {
    // get_functions().find(name);
    if (auto iter = get_functions().find(name); iter != get_functions().end()) {
        return iter->second; // Funciton*
    }
    return nullptr;
}
Function *Module::add_function(bool is_decl, Type *type, const_str_ref name) {
    if (get_function(name)) {
        assert(0); // re-def name
    }
    auto func = new Function(this, type, name);
    // _functions.insert({ name, func });
    _values.emplace_back(func);
    _functions.emplace(name, func);

    return func;
}

// Value *Module::register_val(const_str_ref name) {}
// Value *Module::get_val(const_str_ref name) {}
// Value *Module::add_val(const_str_ref name, Value *addr) {}
} // namespace ir
