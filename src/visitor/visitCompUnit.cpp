#include "visitor/visitor.hpp"

#include <stdbool.h>
namespace sysy {
std::any SysYIRGenerator::visitCompUnit(SysYParser::CompUnitContext* ctx) {
  ir::SymbolTable::ModuleScope scope(mTables);
  // add runtime lib functions
  auto type_i32 = ir::Type::TypeInt32();
  auto type_f32 = ir::Type::TypeFloat32();
  auto type_void = ir::Type::void_type();
  auto type_i32p = ir::Type::TypePointer(type_i32);
  auto type_f32p = ir::Type::TypePointer(type_f32);

  //! 外部函数
  mModule->addFunction(ir::Type::TypeFunction(type_i32, {}), "getint");
  mModule->addFunction(ir::Type::TypeFunction(type_i32, {}), "getch");
  mModule->addFunction(ir::Type::TypeFunction(type_f32, {}), "getfloat");

  mModule->addFunction(ir::Type::TypeFunction(type_i32, {type_i32p}), "getarray");
  mModule->addFunction(ir::Type::TypeFunction(type_i32, {type_f32p}), "getfarray");

  mModule->addFunction(ir::Type::TypeFunction(type_void, {type_i32}), "putint");
  mModule->addFunction(ir::Type::TypeFunction(type_void, {type_i32}), "putch");
  mModule->addFunction(ir::Type::TypeFunction(type_void, {type_f32}), "putfloat");

  mModule->addFunction(ir::Type::TypeFunction(type_void, {type_i32, type_i32p}), "putarray");
  mModule->addFunction(ir::Type::TypeFunction(type_void, {type_i32, type_f32p}), "putfarray");

  mModule->addFunction(ir::Type::TypeFunction(type_void, {}), "putf");

  mModule->addFunction(ir::Type::TypeFunction(type_void, {type_i32}), "starttime");
  mModule->addFunction(ir::Type::TypeFunction(type_void, {type_i32}), "stoptime");

  mModule->addFunction(ir::Type::TypeFunction(type_void, {type_i32}), "_sysy_starttime");
  mModule->addFunction(ir::Type::TypeFunction(type_void, {type_i32}), "_sysy_stoptime");

  // visitChildren(ctx);
  for (auto declCtx : ctx->decl()) {
    visit(declCtx);
  }
  for (auto funcDeclCtx : ctx->funcDecl()) {
    mBuilder = ir::IRBuilder();
    visit(funcDeclCtx);
  }
  for (auto funcDefCtx : ctx->funcDef()) {
    mBuilder = ir::IRBuilder();
    visit(funcDefCtx);
  }

  return nullptr;
}
}  // namespace sysy
