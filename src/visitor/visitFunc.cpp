#include "visitor/visitor.hpp"

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

            //! TODO
            // up to realize array version

            // std::cout << param->getText() << std::endl;
        }
    }
    // any_cast: cast any to whated type
    ir::Type* ret_type = any_cast_Type(visit(ctx->funcType()));

    // empty param types
    ir::Type* func_type = ir::Type::func_type(ret_type, param_types);
    // add func to module
    ir::Function* func = module()->add_func(func_type, func_name);

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

        ir::BasicBlock* entry = func->new_entry();
        ir::BasicBlock* exit = func->new_exit();
        entry->append_comment("entry");
        exit->append_comment("exit");
        builder().set_pos(entry, entry->begin());
        // create return value alloca
        auto fz = ir::Constant::gen_f32(0.0);
        if (not func->ret_type()->is_void()) {
            auto ret_value_ptr = builder().create_alloca(func->ret_type(), {});
            switch (func->ret_type()->btype()) {
                case ir::INT32: 
                    builder().create_store(ir::Constant::gen_i32(0), ret_value_ptr);
                    break;
                case ir::FLOAT:
                    
                    builder().create_store(ir::Constant::gen_f32(0.0), ret_value_ptr);
                    break;
                default: 
                    assert(false && "not valid type");
            }
            // auto ret_store_zero = builder().create_store(ir::Constant::gen_i32(0), ret_value_ptr);
            func->set_ret_value_ptr(ret_value_ptr); 
        }


        if (ctx->funcFParams()){
            // first init args
            for (auto pram : ctx->funcFParams()->funcFParam()) {
                auto arg_name = pram->ID()->getText();
                auto arg_type = any_cast_Type(visit(pram->btype()));
                auto arg = func->new_arg(arg_type);
            }

            // init return value ptr and first block
            entry->set_name(builder().get_bbname());

            // allca all params and store
            int idx = 0;
            for (auto pram : ctx->funcFParams()->funcFParam()) {
                auto arg_name = pram->ID()->getText();
                auto arg_type = any_cast_Type(visit(pram->btype()));
                // no const arg
                auto alloca_ptr = builder().create_alloca(arg_type, {});
                auto store = builder().create_store(func->arg_i(idx), alloca_ptr, "store");
                _tables.insert(arg_name, alloca_ptr);
                idx++;
            }

        }
        else {
            entry->set_name(builder().get_bbname());
        }


        visitBlockStmt(ctx->blockStmt());

        builder().create_br(exit);
        ir::BasicBlock::block_link(builder().block(), exit);
        
        exit->set_name(builder().get_bbname());
        builder().set_pos(exit, exit->begin());

        if (not func->ret_type()->is_void()) {
            auto ret_value = builder().create_load(func->ret_value_ptr(), {});
            builder().create_return(ret_value);
        } else {
            builder().create_return();
        }

        
        func->sort_blocks();
    }

    builder().reset();
    return dyn_cast_Value(func);
}

}  // namespace sysy
