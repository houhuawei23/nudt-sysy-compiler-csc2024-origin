#pragma once
#include "ir/constant.hpp"
#include "ir/function.hpp"
#include "ir/global.hpp"
#include "ir/value.hpp"

namespace ir {
class Module {
 private:
  std::vector<Function*> mFunctions;
  std::unordered_map<std::string, Function*> mFuncTable;

  std::vector<GlobalVariable*> mGlobalVariables;
  std::unordered_map<std::string, GlobalVariable*> mGlobalVariableTable;

 public:
  Module() = default;

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
  void print(std::ostream& os);
};
}