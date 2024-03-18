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
 * @brief
 *
 * @param ctx
 * @return std::any
 *
 * func: funcType ID LPAREN funcFParams? RPAREN blockStmt;
 * ```
 * int func();
 * int func(int a);
 * int func(int a) {
 *  return a + 5;
 * }
 * ```
 */

std::any SysYIRGenerator::visitFunc(SysYParser::FuncContext* ctx) {
    _builder.func_inc();
    std::string func_name = ctx->ID()->getText();
    std::vector<ir::Type*> param_types;
    std::vector<std::string> param_names;
    int length = 0;
    // if have formal params
    if (ctx->funcFParams()) {
        // funcFParams: funcFParam (COMMA funcFParam)*;
        auto params = ctx->funcFParams()->funcFParam();
        // cout << params->getText() << endl;
        // std::cout << typeid(params).name() << std::endl;
        
        for (auto param : params) {
            if(param->btype()->INT())
                param_types.push_back(ir::Type::i32_type());
            else
                param_types.push_back(ir::Type::float_type());
            param_names.push_back(param->ID()->getText());
            length++;
            //!TODO 
            // up to realize array version

            // std::cout << param->getText() << std::endl;
        }
    }
    // any_cast: cast any to whated type
    ir::Type* ret_type = any_cast_Type(visit(ctx->funcType()));

    // empty param types
    ir::Type* func_type = ir::Type::function_type(ret_type, param_types);
    // add func to module
    ir::Function* func = module()->add_function(true, func_type, func_name);

    // this->module()->insert??

    // create and enter function scope
    // it will be automatically destroyed when return from this visitFfunc
    ir::SymbolTableBeta::FunctionScope scope(_tables);
    // create entry block with the same params of func
    ir::BasicBlock* entry = func->new_block();
    entry->set_name(builder().getvarname());
    if (ctx->funcFParams()) {  // has formal params
        //! TODO: add fparams to entry block
        // add insert to symbol table
        for(int i = 0;i < length;i++)
        {//TODO
            auto alloca_ptr = builder().create_alloca(param_types[i],{},_builder.getvarname(),false);
            _tables.insert(param_names[i], alloca_ptr);
        }
        
    }

    _builder.set_pos(entry, entry->end());
    visitBlockStmt(ctx->blockStmt());
    func->sort_blocks();
    return func;
}

}  // namespace sysy
