#include <map>
#include <set>
#include <unordered_map>
// #include <unordered_set>
#include <variant>

#include "infrast.hpp"
#include "table.hpp"
#include "type.hpp"
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
    // using inst_list = std::list<std::unique_ptr<Instruction>>; // list
    // using iterator = inst_list::iterator;
    // using reverse_iterator = inst_list::reverse_iterator;

    // using arg_list = std::list<std::unique_ptr<Argument>>;     // vector ->
    // list using block_list = std::list<std::unique_ptr<BasicBlock>>; // vector
    // -> list

    using value_vector = std::vector<std::unique_ptr<Value>>;
    using str_fun_map = std::map<std::string, Function *>;
    using str_val_map = std::map<std::string, Value *>;
    using const_str_ref = const std::string &;
    // using value
  protected:
    value_vector _values;
    str_fun_map _functions;
    str_val_map _globals;

    SymbolTable _stable;
    int _test;

  public:
    Module(){};
    Module(int test) : _test(test){};
    // ~Module() = default;

    //! get
    // return the ref, avoid generate temp var
    // using original type to recieve, point to new?
    // using ref type to receive, they point to same obj
    // directly using, point to same: get_values().push xxx
    // how about use iterator to access?
    value_vector &get_values() { return _values; }
    str_fun_map &get_functions() { return _functions; }
    str_val_map &get_globals() { return _globals; }

    Function *get_function(const_str_ref name);
    Function *add_function(bool is_decl, Type *type, const_str_ref name);

    // Value *register_val(const_str_ref name);
    // Value *get_val(const_str_ref name);
    // Value *add_val(const_str_ref name, Value *addr);

    void add_gvalue(const_str_ref name, Value*init);
    Value *lookup_gvalue(const_str_ref name);
};
} // namespace ir
