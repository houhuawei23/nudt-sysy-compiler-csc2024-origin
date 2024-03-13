
#pragma once
#include <any>  // any_cast
#include <typeinfo>
#include <vector>

#include "SysYBaseVisitor.h"
#include "ir.hpp"

#include "utils_visit.hpp"

namespace sysy {
class SysYIRGenerator : public SysYBaseVisitor {
   private:
    ir::Module* _module;
    ir::IRBuilder _builder;
    ir::SymbolTableBeta _tables;
    antlr4::ParserRuleContext* _root;

   public:
    SysYIRGenerator(){};
    SysYIRGenerator(ir::Module* module, antlr4::ParserRuleContext* root)
        : _module(module), _root(root) {}
    // 'get' means get the same obj int the class
    // 'copy' means get the copy of obj
    ir::Module* module() { return _module; }
    ir::IRBuilder& builder() { return _builder; }

    void build_ir() { visit(_root); }

    virtual std::any visitCompUnit(SysYParser::CompUnitContext* ctx) override;

    // virtual std::any visitDecl(SysYParser::DeclContext *ctx) override;

    virtual std::any visitFunc(SysYParser::FuncContext* ctx) override;

    virtual std::any visitFuncType(SysYParser::FuncTypeContext* ctx) override;

    virtual std::any visitBlockStmt(SysYParser::BlockStmtContext* ctx) override;

    // visitDecl
    virtual std::any visitDecl(SysYParser::DeclContext* ctx) override;
    std::any visitDeclLocal(SysYParser::DeclContext* ctx);
    std::any visitDeclGlobal(SysYParser::DeclContext* ctx);

    // virtual std::any visitVarDef(SysYParser::VarDefContext* ctx) override;
    void visitVarDef_beta(SysYParser::VarDefContext* ctx,
                          ir::Type* type,
                          bool is_const);

    virtual std::any visitBtype(SysYParser::BtypeContext* ctx) override;
    virtual std::any visitNumberExp(SysYParser::NumberExpContext* ctx) override;
    virtual std::any visitReturnStmt(
        SysYParser::ReturnStmtContext* ctx) override;

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
}  // namespace sysy
