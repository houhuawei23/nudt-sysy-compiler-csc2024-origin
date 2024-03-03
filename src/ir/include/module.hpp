#include "infrast.hpp"
#include <map>
#include <set>
#include <unordered_map>
// #include <unordered_set>
#include <variant>

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
class Module
{
    using inst_list = std::list<std::unique_ptr<Instruction>>; // list
    using iterator = inst_list::iterator;
    using reverse_iterator = inst_list::reverse_iterator;

    using arg_list = std::list<std::unique_ptr<Argument>>;     // vector -> list
    using block_list = std::list<std::unique_ptr<BasicBlock>>; // vector -> list

    // using value
  protected:
    std::vector<std::unique_ptr<Value>> children;
    std::map<std::string, Function*> functions;
    std::map<std::string, Value*> globals;

  public:
    
};
} // namespace ir
