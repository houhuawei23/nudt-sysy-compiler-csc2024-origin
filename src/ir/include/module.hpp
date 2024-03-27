#pragma once

#include "function.hpp"
#include "global.hpp"
#include "value.hpp"

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

    std::vector<ir::Value*> _gvs;
    str_value_map _gv_table;

   public:
    Module() = default;
    ~Module() = default;

    //! get
    Function* lookup_func(const_str_ref name);

    Function* add_function(Type* type, const_str_ref name);

    void add_gvar(const_str_ref name, Value* gv);

    // readable ir print
    void print(std::ostream& os) const;
};
}  // namespace ir
