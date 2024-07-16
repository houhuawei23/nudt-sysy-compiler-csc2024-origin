#pragma once
#include "ir/constant.hpp"
#include "ir/function.hpp"
#include "ir/global.hpp"
#include "ir/value.hpp"

#include "support/arena.hpp"

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
  utils::Arena mArena;
  std::vector<Function*> mFunctions;
  std::unordered_map<std::string, Function*> mFuncTable;

  std::vector<GlobalVariable*> mGlobalVariables;
  std::unordered_map<std::string, GlobalVariable*> mGlobalVariableTable;

 public:
  Module() : mArena{utils::Arena::Source::IR} {};

  //! get
  auto& funcs() const { return mFunctions; }
  auto& globalVars() const { return mGlobalVariables; }

  Function* mainFunction() const { return findFunction("main"); }

  Function* findFunction(const_str_ref name) const;
  Function* addFunction(Type* type, const_str_ref name);
  void delFunction(ir::Function* func);

  void addGlobalVar(const_str_ref name, GlobalVariable* gv);

  void rename();
  // readable ir print
  void print(std::ostream& os) const;
  bool verify(std::ostream& os) const;
};

SYSYC_ARENA_TRAIT(Module, IR);
}  // namespace ir
