#include "include/module.hpp"
#include <cassert>
#include <iostream>
#include "include/utils_ir.hpp"
namespace ir {

void Module::add_gvar(const_str_ref name, Value* gv) {
    auto iter = _gv_table.find(name);  // find the name in _globals
    assert(iter == _gv_table.end() &&
           "Redeclare! global variable already exists");
    _gv_table.emplace(name, gv);
    _gvs.emplace_back(gv);
}

Function* Module::lookup_func(const_str_ref name) {
    auto iter = _func_table.find(name);
    if (iter != _func_table.end()) {
        return iter->second;
    }
    return nullptr;
}
Function* Module::add_function(Type* type, const_str_ref name) {
    // re-def name
    assert(lookup_func(name) == nullptr && "re-add function!");

    ir::Function* func = new Function(type, name, this);

    _func_table.emplace(name, func);
    _funcs.emplace_back(func);
    return func;
}

// readable ir print
void Module::print(std::ostream& os) const {
    //! print all global values
    for (auto gv : _gvs) {
        if (ir::isa<ir::Constant>(gv)) {
            auto res = dyn_cast<ir::Constant>(gv);
            os << res->name() << " = constant " << *(res->type()) << " ";
            if (res->is_i32())
                os << res->i32() << std::endl;
            else
                os << res->f64() << std::endl;
        } else {
            os << *gv << std::endl;
        }
    }

    //! print all functions
    for (auto func : _funcs) {
        os << *func << std::endl;
    }
}

}  // namespace ir
