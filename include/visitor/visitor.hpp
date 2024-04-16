#pragma once
#include <any>
#include <typeinfo>
#include <vector>
#include <sstream>
#include <iomanip>

#include "SysYBaseVisitor.h"
#include "ir/ir.hpp"

#include "support/utils.hpp"

namespace sysy {
class SysYIRGenerator : public SysYBaseVisitor {
   private:
    ir::Module* _module = nullptr;
    ir::IRBuilder _builder;
    ir::SymbolTableBeta _tables;
    antlr4::ParserRuleContext* _root;

    int _d = 0, _n = 0;
    ir::Type* _current_type = nullptr;
    std::vector<int> _path;
    bool _is_alloca = false;

   public:
    SysYIRGenerator(){};
    SysYIRGenerator(ir::Module* module, antlr4::ParserRuleContext* root)
        : _module(module), _root(root) {}

    // 'get' means get the same obj int the class
    // 'copy' means get the copy of obj

    ir::Module* module() { return _module; }

    ir::IRBuilder& builder() { return _builder; }

    ir::Module* build_ir() {
        visit(_root);
        return _module;
    }

    //! Override all visit methods
    virtual std::any visitCompUnit(SysYParser::CompUnitContext* ctx) override;

    //! function
    virtual std::any visitFuncType(SysYParser::FuncTypeContext* ctx) override;

    virtual std::any visitFuncDef(SysYParser::FuncDefContext* ctx) override;

    ir::Function* create_func(SysYParser::FuncDefContext* ctx);

    virtual std::any visitBlockStmt(SysYParser::BlockStmtContext* ctx) override;

    //! visitDecl
    virtual std::any visitDecl(SysYParser::DeclContext* ctx) override;
    ir::Value* visitDeclLocal(SysYParser::DeclContext* ctx);
    ir::Value* visitDeclGlobal(SysYParser::DeclContext* ctx);

    void visitInitValue_Array(SysYParser::InitValueContext* ctx,
                              const int capacity,
                              const std::vector<int> dims,
                              std::vector<ir::Value*>& init);
    // 局部变量
    ir::Value* visitVarDef_local(SysYParser::VarDefContext* ctx,
                                 ir::Type* btype,
                                 bool is_const);
    ir::Value* visitArray_local(SysYParser::VarDefContext* ctx,
                                ir::Type* btype,
                                bool is_const,
                                std::vector<int> dims,
                                int capacity);
    ir::Value* visitScalar_local(SysYParser::VarDefContext* ctx,
                                 ir::Type* btype,
                                 bool is_const);

    // 全局变量
    ir::Value* visitVarDef_global(SysYParser::VarDefContext* ctx,
                                  ir::Type* btype,
                                  bool is_const);
    ir::Value* visitArray_global(SysYParser::VarDefContext* ctx,
                                 ir::Type* btype,
                                 bool is_const,
                                 std::vector<int> dims,
                                 int capacity);
    ir::Value* visitScalar_global(SysYParser::VarDefContext* ctx,
                                  ir::Type* btype,
                                  bool is_const);

    virtual std::any visitBtype(SysYParser::BtypeContext* ctx) override;

    virtual std::any visitLValue(SysYParser::LValueContext* ctx) override;

    //! visit Stmt
    virtual std::any visitReturnStmt(
        SysYParser::ReturnStmtContext* ctx) override;

    virtual std::any visitAssignStmt(
        SysYParser::AssignStmtContext* ctx) override;

    virtual std::any visitIfStmt(SysYParser::IfStmtContext* ctx) override;

    virtual std::any visitWhileStmt(SysYParser::WhileStmtContext* ctx) override;

    virtual std::any visitBreakStmt(SysYParser::BreakStmtContext* ctx) override;

    virtual std::any visitContinueStmt(
        SysYParser::ContinueStmtContext* ctx) override;

    //! visit EXP
    virtual std::any visitVarExp(SysYParser::VarExpContext* ctx) override;

    virtual std::any visitParenExp(SysYParser::ParenExpContext* ctx) override;

    virtual std::any visitNumberExp(SysYParser::NumberExpContext* ctx) override;

    virtual std::any visitUnaryExp(SysYParser::UnaryExpContext* ctx) override;

    virtual std::any visitMultiplicativeExp(
        SysYParser::MultiplicativeExpContext* ctx) override;

    virtual std::any visitAdditiveExp(
        SysYParser::AdditiveExpContext* ctx) override;

    virtual std::any visitRelationExp(
        SysYParser::RelationExpContext* ctx) override;

    virtual std::any visitEqualExp(SysYParser::EqualExpContext* ctx) override;

    virtual std::any visitAndExp(SysYParser::AndExpContext* ctx) override;

    virtual std::any visitOrExp(SysYParser::OrExpContext* ctx) override;

    //! call
    virtual std::any visitCall(SysYParser::CallContext* ctx) override;
};
}  // namespace sysy
