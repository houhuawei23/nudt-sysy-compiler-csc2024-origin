#include "visitor.hpp"

using namespace std;

namespace sysy {
/**
 * @brief
 *
 * @param ctx
 * @return std::any
 *
 * funcType: VOID | INT | FLOAT;
 */
std::any SysYIRGenerator::visitFuncType(SysYParser::FuncTypeContext* ctx) {
    if (ctx->INT()) {
        return ir::Type::i32_type();
    } else if (ctx->FLOAT()) {
        return ir::Type::float_type();
    } else if (ctx->VOID()) {
        return ir::Type::void_type();
    }
    assert(false);
    return 0;
}

/**
 * func: funcType ID LPAREN funcFParams? RPAREN blockStmt;
 * ```
 * int func();
 * int func(int a);
 * int func(int a) {
 *  return a + 5;
 * }
 * ```
 */

ir::Function* SysYIRGenerator::create_func(SysYParser::FuncDefContext* ctx) {
    std::string func_name = ctx->ID()->getText();
    std::vector<ir::Type*> param_types;

    // std::vector<std::string> param_names;
    // param_names.push_back(param->ID()->getText());

    int length = 0;

    if (ctx->funcFParams()) { // if have formal params
        // funcFParams: funcFParam (COMMA funcFParam)*;
        auto params = ctx->funcFParams()->funcFParam();
        // cout << params->getText() << endl;
        // std::cout << typeid(params).name() << std::endl;

        for (auto param : params) {
            if (param->btype()->INT())
                param_types.push_back(ir::Type::i32_type());
            else
                param_types.push_back(ir::Type::float_type());

            length++;
            //! TODO
            // up to realize array version

            // std::cout << param->getText() << std::endl;
        }
    }
    // any_cast: cast any to whated type
    ir::Type* ret_type = any_cast_Type(visit(ctx->funcType()));

    // empty param types
    ir::Type* func_type = ir::Type::function_type(ret_type, param_types);
    // add func to module
    ir::Function* func = module()->add_function(func_type, func_name);

    return func;
}

// funcFParams: funcFParam (COMMA funcFParam)*;
// funcFParam: btype ID (LBRACKET RBRACKET (LBRACKET exp RBRACKET)*)?;
std::any SysYIRGenerator::visitFuncDef(SysYParser::FuncDefContext* ctx) {
    // _builder.func_inc();
    std::string func_name = ctx->ID()->getText();
    ir::Function* func = module()->lookup_func(func_name);
    if (not func) { // not declared
        func = create_func(ctx);
    }
    // is defined
    if (ctx->blockStmt())  
    {

        // create and enter function scope
        // it will be automatically destroyed when return from this visitFfunc
        ir::SymbolTableBeta::FunctionScope scope(_tables);

        // create entry block with the same params of func
        // ir::BasicBlock* entry = func->create_entry();
        ir::BasicBlock* entry = func->new_block();

        builder().set_pos(entry, entry->begin());
        if (ctx->funcFParams()){
            for (auto pram : ctx->funcFParams()->funcFParam()) {
                auto arg_name = pram->ID()->getText();
                auto arg_type = any_cast_Type(visit(pram->btype()));
                auto arg = func->new_arg(arg_type, builder().getvarname());
            }
            entry->set_name(builder().getvarname());
            int idx = 0;
            for (auto pram : ctx->funcFParams()->funcFParam()) {
                auto arg_name = pram->ID()->getText();
                auto arg_type = any_cast_Type(visit(pram->btype()));
                // no const arg
                auto alloca_ptr = builder().create_alloca(arg_type, {}, builder().getvarname(), false);
                auto store = builder().create_store(func->arg_i(idx), alloca_ptr, {}, "store");
                _tables.insert(arg_name, alloca_ptr);
                idx++;
            }

        }
        else {
            entry->set_name(builder().getvarname());
        }


        visitBlockStmt(ctx->blockStmt());
        func->sort_blocks();
    }

    builder().var_reset();
    return func;
}

}  // namespace sysy
