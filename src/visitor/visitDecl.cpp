#include "visitor.hpp"

namespace sysy {
//! btype: INT | FLOAT;
std::any SysYIRGenerator::visitBtype(SysYParser::BtypeContext* ctx) {
    // change Btype in c to Btype in LLVM
    if (ctx->INT()) {
        return ir::Type::i32_type();
    } else if (ctx->FLOAT()) {
        //! transfer all float to double
        return ir::Type::double_type();
        // return ir::Type::float_type();
    }
    // else if (ctx->DOUBLE()) {
    //     return ir::Type::double_type();
    // } else if (ctx->VOID()) {
    //     return ir::Type::void_type();
    // }

    // return ctx->INT() ? ir::Type::i32_type() : ir::Type::float_type();
    return nullptr;
}

std::any SysYIRGenerator::visitDecl(SysYParser::DeclContext* ctx) {
    return _tables.isModuleScope() ? visitDeclGlobal(ctx) : visitDeclLocal(ctx);
}

/*
 * @brief Visit Local Variable Declaration
 *      decl: CONST? btype varDef (COMMA varDef)* SEMICOLON;
 */
std::any SysYIRGenerator::visitDeclLocal(SysYParser::DeclContext* ctx) {
    auto btype = any_cast_Type(visitBtype(ctx->btype()));
    bool is_const = ctx->CONST();
    ir::Value* res = nullptr;

    for (auto varDef : ctx->varDef()) {
        res = visitVarDef_beta(varDef, btype, is_const);
    }
    return res;
}

/*
exp:
    LPAREN exp RPAREN				# parenExp
    | var						# varExp
    | number						# numberExp
    | string						# stringExp
    | call							# callExp
    | (ADD | SUB | NOT) exp			# unaryExp
    | exp (MUL | DIV | MODULO) exp	# multiplicativeExp
    | exp (ADD | SUB) exp			# additiveExp
    | exp (LT | GT | LE | GE) exp	# relationExp
    | exp (EQ | NE) exp				# equalExp
    | exp AND exp					# andExp
    | exp OR exp					# orExp;
*/
/*
 * @brief Visit Local Variable Declaration
 *      decl: CONST? btype varDef (COMMA varDef)* SEMICOLON;
 *      varDef: lValue (ASSIGN initValue)?;
 *      lValue: ID (LBRACKET exp RBRACKET)*;
 * @details
 *      1. 
 */
ir::Value* SysYIRGenerator::visitVarDef_beta(SysYParser::VarDefContext* ctx,
                                             ir::Type* btype,
                                             bool is_const) {
    auto name = ctx->lValue()->ID()->getText();

    // 获得数组各个维度
    std::vector<ir::Value*> dims;
    for (auto dim : ctx->lValue()->exp()) {
        dims.push_back(any_cast_Value(visit(dim)));
    }
    std::vector<int> idx(dims.size());

    ir::Value* res = nullptr;
    ir::Value* init = nullptr;
    ir::Constant* cinit = nullptr;

    std::vector<ir::Value*> arrayInit;

    //! get initial value
    if (ctx->ASSIGN()) {
        if (ctx->initValue()->LBRACE()) {  //! 1. array initial value
            for (auto expr : ctx->initValue()->initValue()){
                arrayInit.push_back(any_cast_Value(visitInitValue(expr)));
            }
        } else {  //! 2. scalar initial value
            init = any_cast_Value(visit(ctx->initValue()->exp()));
            if (cinit = ir::dyn_cast<ir::Constant>(init)) {
                if (btype->is_i32() && cinit->is_float()) {
                    cinit = ir::Constant::gen_i32(cinit->f64());
                } else if (btype->is_float() && cinit->is_i32()) {
                    cinit = ir::Constant::gen_f64(cinit->i32());
                }
            }
        }
    }

    //! deal with assignment
    if (is_const) {  //! 1. const
        if (dims.size() == 0) {  //! 1.1 scalar
            if (not ctx->ASSIGN()) {
                std::cerr << "const without initialization!" << std::endl;
                exit(EXIT_FAILURE);
            } else {
                if (cinit) {
                    _tables.insert(name, cinit);
                    res = cinit;
                } else {
                    std::cerr << "can't use variable initialize const" << std::endl;
                    exit(EXIT_FAILURE);
                }
            }
        }
        else {  //! 1.2 array
            // TODO
        }
    } else {  //! 2. variable
        //! create alloca inst
        auto alloca_ptr = _builder.create_alloca(btype, dims, _builder.getvarname(), is_const);
        _tables.insert(name, alloca_ptr);
        
        //! create store inst
        if (dims.size() == 0) {  //! 2.1 scalar
            if(not ctx->ASSIGN()){
            } else if (init->is_float() && btype->is_i32()) {
                auto ftosi = _builder.create_ftosi(ir::Type::i32_type(), init, _builder.getvarname());
                auto stroe = _builder.create_store(ftosi, alloca_ptr, "store");
            } else if (init->is_i32() && btype->is_float()) {
                auto sitof = _builder.create_sitof(ir::Type::float_type(), init, _builder.getvarname());
                auto store = _builder.create_store(sitof, alloca_ptr, "store");
            } else {
                auto store = _builder.create_store(init, alloca_ptr, "store");
            }
        } else {  //! 2.2 array
            if(ctx->ASSIGN()) {
                int dimensions = dims.size();
                ir::Value* ptr =ir::dyn_cast<ir::Value>(alloca_ptr);
                ir::Type* type = alloca_ptr->base_type();

                // // 获得数组相关索引的地址
                // for (int i = 1; i <= dimensions; i++) {
                //     ptr = ir::dyn_cast<ir::Value>(_builder.create_getelementptr(type, ptr, 
                //                                                                 idx, i, dims, 
                //                                                                 _builder.getvarname(), 1));
                //     type = ptr->type();
                // }
            } else{
                // pass
            }
        }

        res = alloca_ptr;
    }
    return res;
}

// varDef: lValue (ASSIGN initValue)?;
// lValue: ID (LBRACKET exp RBRACKET)*;

ir::Value* SysYIRGenerator::visitVarDef_global(SysYParser::VarDefContext* ctx,
                                             ir::Type* btype,
                                             bool is_const) {

    auto name = ctx->lValue()->ID()->getText();
    // array
    std::vector<ir::Value*> dims;
    for (auto dim : ctx->lValue()->exp()) {
        dims.push_back(any_cast_Value(visit(dim)));
    }
    ir::Value* res = nullptr;
    ir::Value* init = nullptr;
    ir::Constant* cinit = nullptr;


    if (ctx->ASSIGN()) {
        init = any_cast_Value(visit(ctx->initValue()->exp()));
        // initializer element must be a compile-time constant
        assert(ir::isa<ir::Constant>(init) && "global var init val must be Constant");

        cinit = ir::dyn_cast<ir::Constant>(init); 

        if (btype->is_i32() && cinit->is_float()) {
            cinit = ir::Constant::gen_i32(cinit->f64());
        } else if (btype->is_float() && cinit->is_i32()) {
            cinit = ir::Constant::gen_f64(cinit->i32());
        }
    }
    else if(not ctx->ASSIGN()) { // default init is zero
        if (btype->is_i32()) {
            cinit = ir::Constant::gen_i32(0);
        } else if (btype->is_float()) {
            cinit = ir::Constant::gen_f64(0.0);
        }
    }
    
    auto gv = ir::GlobalVariable::gen(btype, dims, cinit, _module, name);
    res = gv;
    if (is_const) {              
        //! 1 decl as const
        // dont insert into globals, do constant spread
        if (dims.size() == 0) {  //! 1.1 标量
            _tables.insert(name, gv);
            module()->add_gvar(name, gv); // 可以通过常量传播消除
        }       // const scalar end
        else {  //! 1.2 数组
            // TODO
        } // const array end
    } // decl as 'const' end 
    else { 
        //! 2 not decl as const
        // insert into globals
        if (dims.size() == 0) {  //! 1.1 标量
            _tables.insert(name, gv);
            module()->add_gvar(name, gv);
            res = gv;
        }       // const scalar end
        else {  //! 1.2 数组
            // TODO
        } // const array end

    }
    return res; // return GlobalVariable

}


// decl: CONST? btype varDef (COMMA varDef)* SEMICOLON;
std::any SysYIRGenerator::visitDeclGlobal(SysYParser::DeclContext* ctx) {
    // std::cout << ctx->getText() << std::endl;
    auto btype = any_cast_Type(visit(ctx->btype()));
    // auto ptr_type = ir::Type::pointer_type(btype);

    bool is_const = ctx->CONST();
    ir::Value* res = nullptr;
    for (auto varDef : ctx->varDef()) {
        res = visitVarDef_global(varDef, btype, is_const);
    }

    return res;
}

}  // namespace sysy