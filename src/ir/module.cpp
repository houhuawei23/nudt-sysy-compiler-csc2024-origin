#include "include/module.hpp"
#include "include/utils.hpp"
#include <cassert>
#include <iostream>
namespace ir {
Function *Module::get_function(const_str_ref name) {
    // get_functions().find(name);
    // if (auto iter = get_functions().find(name); iter !=
    // get_functions().end()) {
    //     return iter->second; // Funciton*
    // }
    return nullptr;
}
Function *Module::add_function(bool is_decl, Type *type, const_str_ref name) {
    if (get_function(name)) {
        assert(0); // re-def name
    }
    ir::Function *func = new Function(this, type, name);
    // auto func = new Function(this, type, name);
    // _functions.insert({ name, func });
    _values.emplace_back(func);
    _functions.emplace(name, func);

    return func;
}

// Value *Module::register_val(const_str_ref name) {}
// Value *Module::get_val(const_str_ref name) {}
// Value *Module::add_val(const_str_ref name, Value *addr) {}

// readable ir print
void Module::print(std::ostream &os) const{
    // print all global values

    // print all functions
    for (auto iter : _functions) {
        auto func = iter.second;
        os << *func << std::endl;
    }
}
} // namespace ir
