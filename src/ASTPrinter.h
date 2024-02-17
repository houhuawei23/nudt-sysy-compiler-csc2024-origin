/*===-------------------------------------------===*/
/* ASTPrinter.h CREATED BY TXS 2024-2-17           */
/*===-------------------------------------------===*/
#pragma once

#include "SysYBaseVisitor.h"

class ASTPrinter : public SysYBaseVisitor {
public:
    std::any visitCompUnit(SysYParser::CompUnitContext *ctx) override;

};