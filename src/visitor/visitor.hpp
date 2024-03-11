
#pragma once
#include "SysYBaseVisitor.h"
// #include "f.hpp"
#include "builder.hpp"
#include "infrast.hpp"
#include "module.hpp"
#include "table_beta.hpp"

namespace sysy {
class SysYIRGenerator : public SysYBaseVisitor {
  private:
    ir::Module *_module;
    ir::IRBuilder _builder;
    ir::SymbolTableBeta _tables;

  public:
    SysYIRGenerator(){};
    SysYIRGenerator(ir::Module *module) : _module(module) {}
    // 'get' means get the same obj int the class
    // 'copy' means get the copy of obj
    ir::Module *get_module() { return _module; }
    ir::IRBuilder &get_builder() { return _builder; }

    virtual std::any visitCompUnit(SysYParser::CompUnitContext *ctx) override;

    // virtual std::any visitDecl(SysYParser::DeclContext *ctx) override;

    virtual std::any visitFunc(SysYParser::FuncContext *ctx) override;

    virtual std::any visitFuncType(SysYParser::FuncTypeContext *ctx) override;

    virtual std::any visitBlockStmt(SysYParser::BlockStmtContext *ctx) override;

    // virtual std::any visitBtype(SysYParser::BtypeContext *ctx) override;
    // visitDecl
    virtual std::any visitDecl(SysYParser::DeclContext *ctx) override;

    std::any visitDeclLocal(SysYParser::DeclContext *ctx);

    std::any visitDeclGlobal(SysYParser::DeclContext *ctx);

    virtual std::any visitBtype(SysYParser::BtypeContext *ctx) override;
    virtual std::any visitNumberExp(SysYParser::NumberExpContext *ctx) override;
    // virtual std::any visitBlockItem(SysYParser::BlockItemContext *ctx)
    // override; virtual std::any visitDecl(SysYParser::DeclContext *ctx)
    // override;

    // // visitVarDef
    // virtual std::any visitVarDef(SysYParser::VarDefContext *ctx) override;

    // // lValue
    // virtual std::any visitLValue(SysYParser::LValueContext *ctx) override;
    // // initVAlue
    // virtual std::any visitInitValue(SysYParser::InitValueContext *ctx)
    // override;
    // // exp

    // // visitNumber
    // virtual std::any visitNumber(SysYParser::NumberContext *ctx) override;
};
} // namespace sysy
