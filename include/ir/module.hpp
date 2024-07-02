#pragma once
#include "ir/constant.hpp"
#include "ir/function.hpp"
#include "ir/global.hpp"
#include "ir/value.hpp"

namespace ir {
class Module {
private:
    std::vector<ir::Function*> _funcs;
    str_fun_map _func_table;

    std::vector<ir::Value*> _gvalues;
    str_value_map _gvalue_table;
public:
    Module() = default;
    ~Module() = default;
public:  // get function
    std::vector<ir::Function*>& funcs() { return _funcs; }
    std::vector<ir::Value*>& gvalues() { return _gvalues; }
public:  // utils function
    Function* lookup_func(const_str_ref name);
    Function* add_func(Type* type, const_str_ref name);
    void add_gvar(const_str_ref name, Value* gv);
    void print(std::ostream& os);
    void rename();
};
}