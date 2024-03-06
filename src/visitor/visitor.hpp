
#pragma once
#include "SysYBaseVisitor.h"
// #include "f.hpp"
#include "builder.hpp"
#include "infrast.hpp"

namespace sysy {
class SysYIRGenerator : public SysYBaseVisitor {
  private:
    ir::Module *_module;
    ir::IRBuilder _builder;

  public:
    SysYIRGenerator(){};
    SysYIRGenerator(ir::Module *module) : _module(module) {}

    virtual std::any visitCompUnit(SysYParser::CompUnitContext *ctx) override;

    virtual std::any visitFunc(SysYParser::FuncContext *ctx) override;

    virtual std::any visitFuncType(SysYParser::FuncTypeContext *ctx) override;

    virtual std::any visitBlockStmt(SysYParser::BlockStmtContext *ctx) override;

    // virtual std::any visitDecl(SysYParser::DeclContext *ctx) override;

    // // visitVarDef
    // virtual std::any visitVarDef(SysYParser::VarDefContext *ctx) override;

    // // lValue
    // virtual std::any visitLValue(SysYParser::LValueContext *ctx) override;
    // // initVAlue
    // virtual std::any visitInitValue(SysYParser::InitValueContext *ctx) override;
    // // exp

    // // visitNumber
    // virtual std::any visitNumber(SysYParser::NumberContext *ctx) override;
};
} // namespace sysy
