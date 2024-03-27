#include "visitor/visitor.hpp"

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
    std::vector<ir::Value*> final_rargs;
    auto iter = func->arg_types().begin();
    if(ctx->funcRParams()){
        for (auto exp : ctx->funcRParams()->exp()) {
        // Type* arg_type = 
            auto rarg = any_cast_Value(visit(exp));
            iter++;
            rargs.push_back(rarg);
        }
    }
    

    assert(func->arg_types().size() == rargs.size() && "size not match!");

    int length = rargs.size();
    for(int i = 0;i < length;i++)
    {
        if(func->arg_types()[i]->is_i32() and rargs[i]->is_float()){
            auto ftosi = _builder.create_ftosi(rargs[i]);
            final_rargs.push_back(ftosi);
        }
        else if(func->arg_types()[i]->is_float() and rargs[i]->is_i32()){
            auto sitof = _builder.create_sitof(rargs[i]);
            final_rargs.push_back(sitof);
        }
        else{
            final_rargs.push_back(rargs[i]);
        }
    }

    ir::Value* inst; //  ir::Value* 接收
    inst = builder().create_call(func, final_rargs);
    return dyn_cast_Value(inst);
}

}  // namespace sysy
