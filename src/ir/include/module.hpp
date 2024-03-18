#pragma once

#include "function.hpp"
#include "value.hpp"
#include "global.hpp"

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
   public:
    value_ptr_vector _values;
    str_fun_map _functions;
    str_value_map _globals;
    // SymbolTable _stable;

   public:
    Module() = default;
    ~Module() = default;

    //! get
    // return the ref, avoid generate temp var
    // using original type to recieve, point to new?
    // using ref type to receive, they point to same obj
    // directly using, point to same: values().push xxx
    // how about use iterator to access?
    value_ptr_vector& values() { return _values; }
    str_fun_map& functions() { return _functions; }
    str_value_map& globals() { return _globals; }

    Function* function(const_str_ref name);
    Function* add_function(bool is_decl, Type* type, const_str_ref name);

    void add_gvar(const_str_ref name, GlobalVariable* gv) {
        auto iter  = _globals.find(name); // find the name in _globals
        assert(iter == _globals.end() && "Redeclare! global variable already exists");
        _globals.emplace(name, gv);
    }

    // Value *register_val(const_str_ref name);
    // Value *get_val(const_str_ref name);
    // Value *add_val(const_str_ref name, Value *addr);

    // void add_gvalue(const_str_ref name, Value*init);
    // Value *lookup_gvalue(const_str_ref name);

    // readable ir print
    void print(std::ostream& os) const;
};
}  // namespace ir
