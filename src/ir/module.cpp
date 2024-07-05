#include "ir/module.hpp"
#include <cassert>
#include <iostream>
#include <algorithm>
// #include "ir/constant.hpp"
#include "ir/utils_ir.hpp"
namespace ir {

void Module::addGlobalVar(const_str_ref name, GlobalVariable* gv) {
  auto iter = mGlobalVariableTable.find(name);  // find the name in _globals
  assert(iter == mGlobalVariableTable.end() &&
         "Redeclare! global variable already exists");
  mGlobalVariableTable.emplace(name, gv);
  mGlobalVariables.emplace_back(gv);
}

Function* Module::findFunction(const_str_ref name) const {
  auto iter = mFuncTable.find(name);
  if (iter != mFuncTable.end()) {
    return iter->second;
  }
  return nullptr;
}
Function* Module::addFunction(Type* type, const_str_ref name) {
  // re-def name
  assert(findFunction(name) == nullptr && "re-add function!");

  ir::Function* func = new Function(type, name, this);

  mFuncTable.emplace(name, func);
  mFunctions.emplace_back(func);
  return func;
}

void Module::delFunction(ir::Function* func) {
  assert(findFunction(func->name()) != nullptr && "delete unexisted function!");
  mFuncTable.erase(func->name());
  mFunctions.erase(std::find(mFunctions.begin(), mFunctions.end(), func));
  // for (auto bb : func->blocks()) {
  //     func->forceDelBlock(bb);
  // }
}

// readable ir print
void Module::print(std::ostream& os) {
  //! print all global values
  // rename();
  for (auto gv : globalVars()) {
    os << *gv << std::endl;
  }

  // llvm inline function
  os << "declare void @llvm.memset.p0i8.i64(i8* nocapture writeonly, i8, "
        "i64, i1 immarg)"
     << std::endl;

  //! print all functions
  for (auto func : mFunctions) {
    os << *func << std::endl;
  }
}

void Module::rename() {
  for (auto func : mFunctions) {
    func->rename();
  }
}

}  // namespace ir
