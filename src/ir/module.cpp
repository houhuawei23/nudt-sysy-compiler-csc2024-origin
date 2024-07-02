#include "ir/module.hpp"
#include <cassert>
#include <iostream>
#include "ir/utils_ir.hpp"
namespace ir {
void Module::add_gvar(const_str_ref name, Value* gv) {
    auto iter = _gvalue_table.find(name);
    assert(iter == _gvalue_table.end() && "Redeclare! global variable already exists");
    _gvalue_table.emplace(name, gv);
    _gvalues.emplace_back(gv);
}
Function* Module::lookup_func(const_str_ref name) {
    auto iter = _func_table.find(name);
    if (iter != _func_table.end()) return iter->second;
    return nullptr;
}
Function* Module::add_func(Type* type, const_str_ref name) {
    assert(lookup_func(name) == nullptr && "re-add function!");
    ir::Function* func = new Function(type, name, this);
    _func_table.emplace(name, func);
    _funcs.emplace_back(func);
    return func;
}
void Module::print(std::ostream& os) {
    rename();
    //! 1. print all global values
    for (auto gv : gvalues()) {
        if (ir::isa<ir::Constant>(gv)) {
            auto res = dyn_cast<ir::Constant>(gv);
            os << res->name() << " = constant " << *(res->type()) << " ";
            if (res->is_i32()) os << res->i32() << std::endl;
            else os << res->f32() << std::endl;
        } else {
            os << *gv << std::endl;
        }
    }
    //! 2. print llvm inline function
    os << "declare void @llvm.memset.p0i8.i64(i8* nocapture writeonly, i8, i64, i1 immarg)\n\n";
    //! 3. print all functions
    for (auto func : _funcs) os << *func << std::endl;
}
void Module::rename() { for (auto func : _funcs) func->rename(); }
}