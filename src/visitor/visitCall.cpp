#include "visitor.hpp"

namespace sysy {

/*

call: ID LPAREN funcRParams? RPAREN;

funcRParams: exp (COMMA exp)*;

var: ID (LBRACKET exp RBRACKET)*;

lValue: ID (LBRACKET exp RBRACKET)*;

*/
std::any SysYIRGenerator::visitCall(SysYParser::CallContext* ctx) {
    auto func_name = ctx->ID()->getText();
    auto func = module()->lookup_func(func_name);
    auto parent_func = builder().block()->parent();

    std::vector<ir::Value*> rargs;

    auto iter = func->arg_types().begin();
    for (auto exp : ctx->funcRParams()->exp()) {
        // Type* arg_type = 
        auto rarg = any_cast_Value(visit(exp));
        if(ir::isa<ir::Constant>(rarg)) {

        }
        else {

        }

        iter++;
        rargs.push_back(rarg);

    }
    ir::Value* inst; //  ir::Value* 接收
    inst = builder().create_call(func, rargs, builder().getvarname());

    return inst;
}

}  // namespace sysy
