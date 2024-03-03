/*===-------------------------------------------===*/
/* ASTPrinter.cpp CREATED BY TXS 2024-2-17         */
/*===-------------------------------------------===*/
#include <algorithm>
#include <iostream>
using namespace std;
// using namespace antlr4;
#include "ASTPrinter.h"
#include "SysYParser.h"
// #include "Token.h"

any ASTPrinter::visitCompUnit(SysYParser::CompUnitContext *ctx)
{
  // cout << ctx->CompUnit()->getText();
  cout << "visitCompUnit" << endl;
  cout << ctx->getText() << endl;
  // cout << ctx->getStart()->getText() << endl;
  // cout << ctx->getStop()->getText() << endl;
  antlr4::Token* start = ctx->getStart();
  cout << start->getText() << endl;
  cout << start->getLine() << endl;
  cout << start->getCharPositionInLine() << endl;
  cout << start->getChannel() << endl;
  cout << start->getTokenIndex() << endl;
  cout << start->getStartIndex() << endl;
  cout << start->getStopIndex() << endl;
  // Parser
  // int *p;
  return nullptr;
}

// any ASTPrinter::visitNumber(SysYParser::NumberContext *ctx) {
//   cout << ctx->IntConst()->getText();
//   return nullptr;
// }

// any ASTPrinter::visitString(SysYParser::StringContext *ctx) {
//   cout << ctx->String()->getText();
//   return nullptr;
// }

// any ASTPrinter::visitFuncRParams(SysYParser::FuncRParamsContext *ctx) {
//   if (ctx->funcRParam().empty())
//     return nullptr;
//   auto numParams = ctx->funcRParam().size();
//   ctx->funcRParam(0)->accept(this);
//   for (int i = 1; i < numParams; ++i) {
//     cout << ", ";
//     ctx->funcRParam(i)->accept(this);
//   }
//   cout << '\n';
//   return nullptr;
// }
