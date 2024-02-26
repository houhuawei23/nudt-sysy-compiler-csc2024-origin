/*===-------------------------------------------===*/
/* ASTPrinter.h CREATED BY TXS 2024-2-17           */
/*===-------------------------------------------===*/
#pragma once

#include "SysYBaseVisitor.h"

class ASTPrinter : public SysYBaseVisitor
{
public:
    std::any visitCompUnit(SysYParser::CompUnitContext *ctx) override;
    // std::any visitDecl(SysYParser::DeclContext *ctx) override;
    // std::any visitNumber(SysYParser::NumberContext *ctx) override;

    // std::any visitFuncRParams(SysYParser::FuncRParamsContext *ctx) override;
    //   std::any visitExpAsRParam(SysYParser::ExpAsRParamContext *ctx) override;
    //   std::any visitStringAsRParam(SysYParser::StringAsRParamContext *ctx) override;
    //   std::any visitString(SysYParser::StringContext *ctx) override;
};