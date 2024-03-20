#include "visitor.hpp"

namespace sysy {
/*
 * @brief Visit Local Variable Declaration
 *      btype: INT | FLOAT;
 */
std::any SysYIRGenerator::visitBtype(SysYParser::BtypeContext* ctx) {
    if (ctx->INT()) {
        return ir::Type::i32_type();
    } else if (ctx->FLOAT()) {
        return ir::Type::double_type();
    }
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
                // arrayInit.push_back(any_cast_Value(visitInitValue(expr)));
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
            } else{
                // pass
            }
        }

        res = alloca_ptr;
    }
    return res;
}

/*
 * @brief Visit Global Variable Declaration
 *      decl: CONST? btype varDef (COMMA varDef)* SEMICOLON;
 *      varDef: lValue (ASSIGN initValue)?;
 *      lValue: ID (LBRACKET exp RBRACKET)*;
 */
ir::Value* SysYIRGenerator::visitVarDef_global(SysYParser::VarDefContext* ctx,
                                               ir::Type* btype, bool is_const) {
    auto name = ctx->lValue()->ID()->getText();
    
    // 获得数组的各个维度
    std::vector<ir::Value*> dims;
    for (auto dim : ctx->lValue()->exp()) {
        ir::Value* tmp = any_cast_Value(visit(dim));
        if (auto ctmp = ir::dyn_cast<ir::Constant>(tmp)) {
            if (ctmp->is_float()) tmp = ir::Constant::gen_i32(ctmp->f64());
        } else {
            assert(false && "dimension must be a constant");
        }
        dims.push_back(tmp);
    }
    bool isArray = dims.size() > 0;

    ir::Value* res = nullptr;
    std::vector<ir::Constant*> init;
    ir::Constant* cinit;


    if (ctx->ASSIGN()) {
        if (isArray) {  //! 1. array initial value
            int capacity = 1;
            for (auto dim : dims) {
                capacity *= ir::dyn_cast<ir::Constant>(dim)->i32();
            } 
            for (int i = 0; i < capacity; i++) {
                init.push_back(ir::Constant::gen_i32(0));
            }

            _d = 0; _n = 0;
            _path.clear(); _path = std::vector<int>(dims.size(), 0);
            _current_type = btype; _is_alloca = false;
            
            for (auto expr : ctx->initValue()->initValue()) {
                visitInitValue_beta(expr, capacity, dims, init);
            }
        } else {  //! 2. scalar initial value
            ir::Value* tmp = any_cast_Value(visit(ctx->initValue()->exp()));
            assert(ir::isa<ir::Constant>(tmp) && "global var init val must be Constant");

            cinit = ir::dyn_cast<ir::Constant>(tmp);
            cinit->set_name("@" + name);
            if (btype->is_i32() && cinit->is_float()) {
                cinit = ir::Constant::gen_i32(cinit->f64(), "@" + name);
            } else if (btype->is_float() && cinit->is_i32()) {
                cinit = ir::Constant::gen_f64(cinit->i32(), "@" + name);
            }
            init.push_back(cinit);
        }
    } else {  // note: default init is zero
        if (isArray) {  //! 1. array initial value
            // pass
        } else {  //! 2. scalar initial value
            if (btype->is_i32()) {
                cinit = ir::Constant::gen_i32(0, "@" + name);
            } else if (btype->is_float()) {
                cinit = ir::Constant::gen_f64(0.0, "@" + name);
            } else {
                assert(false && "invalid type");
            }
            init.push_back(cinit);
        }
    }
    
    if (is_const) {  //! 1. const
        if (dims.size() == 0) {  //! 1.1 标量
            res = cinit;
            _tables.insert(name, cinit);
            module()->add_gvar(name, cinit);
        }
        else {  //! 1.2 数组
            // TODO: 待完善 (一些细节问题)
            auto gv = ir::GlobalVariable::gen(btype, init, dims, _module, "@" + name);
            _tables.insert(name, gv);
            module()->add_gvar(name, gv);
            res = gv;
        }
    } else {  //! 2. variable (数组 OR 变量)
        auto gv = ir::GlobalVariable::gen(btype, init, dims, _module, "@" + name);
        _tables.insert(name, gv);
        module()->add_gvar(name, gv);
        res = gv;
    }
    return res;
}


// decl: CONST? btype varDef (COMMA varDef)* SEMICOLON;
std::any SysYIRGenerator::visitDeclGlobal(SysYParser::DeclContext* ctx) {
    auto btype = any_cast_Type(visit(ctx->btype()));

    bool is_const = ctx->CONST();
    ir::Value* res = nullptr;
    for (auto varDef : ctx->varDef()) {
        res = visitVarDef_global(varDef, btype, is_const);
    }

    return res;
}

/*
 * @brief visit initvalue
 *      varDef: lValue (ASSIGN initValue)?;
 *      initValue: exp | LBRACE (initValue (COMMA initValue)*)? RBRACE;
 */
void SysYIRGenerator::visitInitValue_beta(SysYParser::InitValueContext *ctx, 
                                              const int capacity, const std::vector<ir::Value*> dims, 
                                              std::vector<ir::Constant*>& init) {
    if (ctx->exp()) {
        auto value = any_cast_Value(visit(ctx->exp()));

        if (auto cvalue = ir::dyn_cast<ir::Constant>(value)) {  //! 1. 常量
            if (_current_type->is_i32() && cvalue->is_float()) {
                value = ir::Constant::gen_i32((int)cvalue->f64());
            } else if (_current_type->is_float() && cvalue->is_i32()) {
                value = ir::Constant::gen_f64((float)cvalue->i32());
            }
        } else {  //! 2. 变量
            if (_current_type->is_i32() && value->is_float()) {
                value = _builder.create_ftosi(ir::Type::i32_type(), value, _builder.getvarname());
            } else if (_current_type->is_float() && value->is_i32()) {
                value = _builder.create_sitof(ir::Type::float_type(), value, _builder.getvarname());
            }
        }

        // goto the last dimension
        while (_d < dims.size() - 1) {
            _path[_d++] = _n;
            _n = 0;
        }
        std::vector<ir::Value*> indices;  // 大小为数组维度
        for (int i = 0; i < dims.size() - 1; i++) {
            indices.push_back(ir::Constant::gen_i32(_path[i]));
        }
        indices.push_back(ir::Constant::gen_i32(_n));

        if (_is_alloca) {  //! 局部变量
        } else {  //! 全局变量
            if (auto cvalue = ir::dyn_cast<ir::Constant>(value)) {
                if (cvalue->is_i32()) {
                    int i32 = cvalue->i32();
                    int factor = 1, offset = 0;
                    for (int i = indices.size() - 1; i >= 0; i--) {
                        offset += factor * ir::dyn_cast<ir::Constant>(indices[i])->i32();
                        factor *= ir::dyn_cast<ir::Constant>(dims[i])->i32();
                    }
                    init[offset] = ir::Constant::gen_i32(cvalue->i32());
                } else if (cvalue->is_float()) {
                    float f = cvalue->f64();
                } else {
                    assert(false && "invalid type");
                }
            } else {
                assert(false && "global variable must be initialized by constant");
            }

            // goto next element
            _n++;
            while (_d >= 0 && _n >= ir::dyn_cast<ir::Constant>(dims[_d])->i32()) {
                _n = _path[--_d] + 1;
            }
        }
    } else {
        int cur_d = _d, cur_n = _n;
        for (auto expr : ctx->initValue()) {
            visitInitValue_beta(expr, capacity, dims, init);
        }
        _d = cur_d, _n = cur_n; _n++;

        if (_is_alloca){  //! 局部变量
        } else {  //! 全局变量
            while (_d >= 0 && _n >= ir::dyn_cast<ir::Constant>(dims[_d])->i32()) {
                _n = _path[--_d] + 1;
            }
        }
    }
}

}  // namespace sysy