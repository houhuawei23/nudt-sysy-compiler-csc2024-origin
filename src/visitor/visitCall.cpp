#include "visitor/visitor.hpp"

namespace sysy {
/*
 * @brief: visit call
 * @details: 
 *      call: ID LPAREN funcRParams? RPAREN;
 *      funcRParams: exp (COMMA exp)*;
 *      var: ID (LBRACKET exp RBRACKET)*;
 *      lValue: ID (LBRACKET exp RBRACKET)*;
 */
std::any SysYIRGenerator::visitCall(SysYParser::CallContext* ctx) {
    auto func_name = ctx->ID()->getText();
    /* macro replace */
    if (func_name.compare("starttime") == 0) {
        func_name = "_sysy_starttime";
    } else if (func_name.compare("stoptime") == 0) {
        func_name = "_sysy_stoptime";
    }
    auto func = module()->lookup_func(func_name);
    auto parent_func = builder().block()->parent();
    // function rargs 应该被作为 function 的 operands
    std::vector<ir::Value*> rargs;
    std::vector<ir::Value*> final_rargs;
    auto iter = func->arg_types().begin();
    if (ctx->funcRParams()) {
        for (auto exp : ctx->funcRParams()->exp()) {
            auto rarg = any_cast_Value(visit(exp));
            iter++;
            rargs.push_back(rarg);
        }
    }
    assert(func->arg_types().size() == rargs.size() && "size not match!");

    int length = rargs.size();
    for (int i = 0;i < length;i++) {
        if (func->arg_types()[i]->is_i32() && rargs[i]->is_float()){
            auto ftosi = _builder.create_unary_beta(ir::ValueId::vFPTOSI, rargs[i], ir::Type::i32_type());
            final_rargs.push_back(ftosi);
        }
        else if (func->arg_types()[i]->is_float() && rargs[i]->is_i32()){
            auto sitof = builder().create_unary_beta(ir::ValueId::vSITOFP, rargs[i], ir::Type::float_type());
            final_rargs.push_back(sitof);
        }
        else{
            final_rargs.push_back(rargs[i]);
        }
    }

    ir::Value* inst = builder().create_call(func, final_rargs);
    return dyn_cast_Value(inst);
}
}  // namespace sysy
