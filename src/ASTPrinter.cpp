/*===-------------------------------------------===*/
/* ASTPrinter.cpp CREATED BY TXS 2024-2-17         */
/*===-------------------------------------------===*/
#include <algorithm>
#include <iostream>
using namespace std;
#include "ASTPrinter.h"
#include "SysYParser.h"

any ASTPrinter::visitCompUnit(SysYParser::CompUnitContext *ctx){
    cout<<"Hello this is func visit compunit"<<endl;
    cout<<ctx->getText()<<endl;
    return visitChildren(ctx);
}