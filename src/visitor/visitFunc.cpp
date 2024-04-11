#include "visitor/visitor.hpp"

using namespace std;

namespace sysy {
/*
 * @brief: visit function type
 * @details: 
 *      funcType: VOID | INT | FLOAT;
 */
std::any SysYIRGenerator::visitFuncType(SysYParser::FuncTypeContext* ctx) {
    if (ctx->INT()) {
        return ir::Type::i32_type();
    } else if (ctx->FLOAT()) {
        return ir::Type::float_type();
    } else if (ctx->VOID()) {
        return ir::Type::void_type();
    } else {
        assert(false && "invalid return type");
    }
}

/*
 * @brief: create function
 * @details: 
 *      funcDef: funcType ID LPAREN funcFParams? RPAREN blockStmt;
 *      funcFParams: funcFParam (COMMA funcFParam)*;
 *      funcFParam: btype ID (LBRACKET RBRACKET (LBRACKET exp RBRACKET)*)?;
 */
ir::Function* SysYIRGenerator::create_func(SysYParser::FuncDefContext* ctx) {
    std::string func_name = ctx->ID()->getText();
    std::vector<ir::Type*> param_types;

    if (ctx->funcFParams()) {
        auto params = ctx->funcFParams()->funcFParam();

        for (auto param : params) {
            bool isArray = not param->LBRACKET().empty();

            ir::Type* base_type = any_cast_Type(visit(param->btype()));
            
            if (!isArray) {
                param_types.push_back(base_type);
            } else {
                std::vector<int> dims;
                for (auto expr : param->exp()) {
                    auto value = any_cast_Value(visit(expr));
                    if (auto cvalue = dyn_cast<ir::Constant>(value)) {
                        if (cvalue->is_float()) dims.push_back((int)cvalue->f32());
                        else dims.push_back(cvalue->i32());
                    } else {
                        assert(false && "function parameter must be constant");
                    }
                }

                if (dims.size() != 0) base_type = ir::Type::array_type(base_type, dims);
                param_types.push_back(ir::Type::pointer_type(base_type));
            }
        }
    }

    ir::Type* ret_type = any_cast_Type(visit(ctx->funcType()));
    ir::Type* func_type = ir::Type::func_type(ret_type, param_types);
    ir::Function* func = module()->add_func(func_type, func_name);

    return func;
}

/*
 * @brief: visit function define
 * @details: 
 *      funcDef: funcType ID LPAREN funcFParams? RPAREN blockStmt;
 *      funcFParams: funcFParam (COMMA funcFParam)*;
 *      funcFParam: btype ID (LBRACKET RBRACKET (LBRACKET exp RBRACKET)*)?;
 */
std::any SysYIRGenerator::visitFuncDef(SysYParser::FuncDefContext* ctx) {
    std::string func_name = ctx->ID()->getText();
    ir::Function* func = module()->lookup_func(func_name);
    if (not func) func = create_func(ctx);
    
    if (ctx->blockStmt()) {

        // create and enter function scope
        // it will be automatically destroyed when return from this visitFfunc
        ir::SymbolTableBeta::FunctionScope scope(_tables);

        ir::BasicBlock* entry = func->new_entry();
        ir::BasicBlock* exit = func->new_exit();
        entry->append_comment("entry");
        exit->append_comment("exit");
        auto next = func->new_block();
  
        entry->set_name(builder().get_bbname());

        builder().set_pos(next, next->begin());
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

            func->set_ret_value_ptr(ret_value_ptr); 
        }

        if (ctx->funcFParams()){
            // first init args
            for (auto pram : ctx->funcFParams()->funcFParam()) {
                auto arg_name = pram->ID()->getText();

                bool isArray = not pram->LBRACKET().empty();
                auto arg_type = any_cast_Type(visit(pram->btype()));
                if (isArray) {
                    std::vector<int> dims;
                    for (auto expr : pram->exp()) {
                        auto value = any_cast_Value(visit(expr));
                        if (auto cvalue = dyn_cast<ir::Constant>(value)) {
                            if (cvalue->is_float()) dims.push_back((int)cvalue->f32());
                            else dims.push_back(cvalue->i32());
                        } else {
                            assert(false && "function parameter must be constant");
                        }
                    }
                    if (dims.size() != 0) arg_type = ir::Type::array_type(arg_type, dims);
                    arg_type = ir::Type::pointer_type(arg_type);
                }

                auto arg = func->new_arg(arg_type);
            }

            // init return value ptr and first block
            next->set_name(builder().get_bbname());

            // allca all params and store
            int idx = 0;
            for (auto pram : ctx->funcFParams()->funcFParam()) {
                auto arg_name = pram->ID()->getText();

                bool isArray = not pram->LBRACKET().empty();
                auto arg_type = any_cast_Type(visit(pram->btype()));
                if (isArray) {
                    std::vector<int> dims;
                    for (auto expr : pram->exp()) {
                        auto value = any_cast_Value(visit(expr));
                        if (auto cvalue = dyn_cast<ir::Constant>(value)) {
                            if (cvalue->is_float()) dims.push_back((int)cvalue->f32());
                            else dims.push_back(cvalue->i32());
                        } else {
                            assert(false && "function parameter must be constant");
                        }
                    }
                    if (dims.size() != 0) arg_type = ir::Type::array_type(arg_type, dims);
                    arg_type = ir::Type::pointer_type(arg_type);
                }

                auto alloca_ptr = builder().create_alloca(arg_type);
                auto store = builder().create_store(func->arg_i(idx), alloca_ptr);
                _tables.insert(arg_name, alloca_ptr);
                idx++;
            }

        }
        else {
            next->set_name(builder().get_bbname());
        }


        visitBlockStmt(ctx->blockStmt());

        builder().create_br(exit);
        ir::BasicBlock::block_link(builder().block(), exit);
        
        exit->set_name(builder().get_bbname());
        builder().set_pos(exit, exit->begin());

        if (not func->ret_type()->is_void()) {
            auto ret_value = builder().create_load(func->ret_value_ptr());
            builder().create_return(ret_value);
        } else {
            builder().create_return();
        }
        // for entry to next
        builder().set_pos(entry, entry->begin());
        builder().create_br(next);

        func->sort_blocks();
        // func->add_allocas_to_entry();
    }

    builder().reset();
    return dyn_cast_Value(func);
}

}  // namespace sysy
