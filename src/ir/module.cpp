#include "include/module.hpp"
#include "include/utils.hpp"
#include <cassert>
#include <iostream>
namespace ir {
Function *Module::function(const_str_ref name) {
    // functions().find(name);
    // if (auto iter = functions().find(name); iter !=
    // functions().end()) {
    //     return iter->second; // Funciton*
    // }
    return nullptr;
}
Function *Module::add_function(bool is_decl, Type *type, const_str_ref name) {
    if (function(name)) {
        assert(0); // re-def name
    }
    ir::Function *func = new Function(type, name, this);
    
    // _functions.insert({ name, func });
    _values.emplace_back(func);
    _functions.emplace(name, func);

    return func;
}

// Value *Module::register_val(const_str_ref name) {}
// Value *Module::val(const_str_ref name) {}
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
