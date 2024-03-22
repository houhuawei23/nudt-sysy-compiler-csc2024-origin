#include "include/module.hpp"
#include <cassert>
#include <iostream>
#include "include/utils_ir.hpp"
namespace ir {
Function* Module::lookup_func(const_str_ref name) {
    auto iter = _functions.find(name);
    if (iter != _functions.end()) {
        return iter->second;
    }
    return nullptr;
}
Function* Module::add_function(Type* type,
                               const_str_ref name) {
    assert(lookup_func(name) == nullptr && "re-add function!");  // re-def name

    ir::Function* func = new Function(type, name, this);

    _values.emplace_back(func);
    _functions.emplace(name, func);

    return func;
}

// readable ir print
void Module::print(std::ostream& os) const {
    //! print all global values
    for (auto gv_iter : _globals) {
        if (ir::isa<ir::Constant>(gv_iter.second)) {
            auto res = dyn_cast<ir::Constant>(gv_iter.second);
            os << res->name() << " = constant " << *(res->type()) << " ";
            if (res->is_i32()) os << res->i32() << std::endl;
            else os << res->f64() << std::endl;
        } else {
            os << *gv_iter.second << std::endl;
        }
    }

    //! print all functions
    for (auto iter : _functions) {
        auto func = iter.second;
        os << *func << std::endl;
    }
}
}  // namespace ir
