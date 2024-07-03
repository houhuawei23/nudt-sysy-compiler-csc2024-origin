#pragma once
#include "ir/constant.hpp"
#include "ir/function.hpp"
#include "ir/global.hpp"
#include "ir/value.hpp"

namespace ir {
//! IR Unit for representing a SysY compile unit
/**
 * @brief
 *
 *  LLVM IR Module
 * The Module class represents the top level structure present in LLVM programs.
 * An LLVM module is effectively either a translation unit of the original
 * program or a combination of several translation units merged by the linker.
 * The Module class keeps track of a list of Functions, a list of
 * GlobalVariables, and a SymbolTable. Additionally, it contains a few helpful
 * member functions that try to make common operations easy.
 *
 */
class Module {
   private:
    std::vector<ir::Function*> _funcs;
    str_fun_map _func_table;

    std::vector<ir::Value*> _gvalues;
    str_value_map _gvalue_table;

   public:
    Module() = default;
    ~Module() = default;

    //! get
    auto& funcs() const { return _funcs; }
    auto& gvalues() const { return _gvalues; }

    Function* lookup_func(const_str_ref name);

    Function* add_func(Type* type, const_str_ref name);

    Function* main_func() { return lookup_func("main"); }

    void add_gvar(const_str_ref name, Value* gv);

    void delete_func(ir::Function* func);

    // readable ir print
    void print(std::ostream& os);

    void rename();
};
}  // namespace ir
