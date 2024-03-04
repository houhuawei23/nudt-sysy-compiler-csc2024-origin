
#pragma once
#include "SysYBaseVisitor.h"
// #include "f.hpp"
#include "builder.hpp"
#include "infrast.hpp"

namespace sysy {
class SysYIRGenerator : public SysYBaseVisitor
{
  private:
    ir::Module* _module;
    ir::IRBuilder _builder;

  public:
    SysYIRGenerator(){};
    SysYIRGenerator(ir::Module* module)
      : _module(module)
    {
    }

    virtual std::any visitCompUnit(SysYParser::CompUnitContext* ctx) override;

    virtual std::any visitFunc(SysYParser::FuncContext* ctx) override;

    virtual std::any visitFuncType(SysYParser::FuncTypeContext* ctx) override;

    virtual std::any visitBlockStmt(SysYParser::BlockStmtContext* ctx) override;
};
}
