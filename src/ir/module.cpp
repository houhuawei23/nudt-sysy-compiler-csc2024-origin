#include "ir/module.hpp"
#include <cassert>
#include <iostream>
#include <algorithm>
// #include "ir/constant.hpp"
#include "ir/utils_ir.hpp"
namespace ir {

void Module::add_gvar(const_str_ref name, Value* gv) {
    auto iter = _gvalue_table.find(name);  // find the name in _globals
    assert(iter == _gvalue_table.end() && "Redeclare! global variable already exists");
    _gvalue_table.emplace(name, gv);
    _gvalues.emplace_back(gv);
}

Function* Module::lookup_func(const_str_ref name) {
    auto iter = _func_table.find(name);
    if (iter != _func_table.end()) {
        return iter->second;
    }
    return nullptr;
}
Function* Module::add_func(Type* type, const_str_ref name) {
    // re-def name
    assert(lookup_func(name) == nullptr && "re-add function!");

    ir::Function* func = new Function(type, name, this);

    _func_table.emplace(name, func);
    _funcs.emplace_back(func);
    return func;
}

void Module::delete_func(ir::Function* func){
    assert(lookup_func(func->name()) != nullptr && "delete unexisted function!");
    _func_table.erase(func->name());
    _funcs.erase(std::find(_funcs.begin(),_funcs.end(),func));
    for(auto bb:func->blocks()){
        func->force_delete_block(bb);
    }
}


// readable ir print
void Module::print(std::ostream& os) {
    //! print all global values
    rename();
    for (auto gv : gvalues()) {
        if (ir::isa<ir::Constant>(gv)) {
            auto res = dyn_cast<ir::Constant>(gv);
            os << res->name() << " = constant " << *(res->type()) << " ";
            if (res->is_i32())
                os << res->i32() << std::endl;
            else
                os << res->f32() << std::endl;
        } else {
            os << *gv << std::endl;
        }
    }

    // llvm inline function
    os << "declare void @llvm.memset.p0i8.i64(i8* nocapture writeonly, i8, i64, i1 immarg)" << std::endl;

    //! print all functions
    for (auto func : _funcs) {
        os << *func << std::endl;
    }
}

void Module::rename(){
    for(auto func : _funcs){
        func->rename();
    }
}

}  // namespace ir
