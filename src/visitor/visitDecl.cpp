#include "visitor.hpp"

namespace sysy {
/*
 * @brief Visit Variable Type
 * @details
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


/*
 * @brief Visit Variable Declaration (变量定义 && 声明)
 */
std::any SysYIRGenerator::visitDecl(SysYParser::DeclContext* ctx) {
    if (_tables.isModuleScope()) {
        return visitDeclGlobal(ctx);
    } else {
        return visitDeclLocal(ctx);
    }
}

/*
 * @brief Visit Local Variable Declaration (局部变量定义 && 声明)
 * @details
 *      decl: CONST? btype varDef (COMMA varDef)* SEMICOLON;
 */
ir::Value* SysYIRGenerator::visitDeclLocal(SysYParser::DeclContext* ctx) {
    auto btype = any_cast_Type(visitBtype(ctx->btype()));
    bool is_const = ctx->CONST();
    ir::Value* res = nullptr;

    for (auto varDef : ctx->varDef()) {
        res = visitVarDef_beta(varDef, btype, is_const);
    }
    return dyn_cast_Value(res);
}

/*
 * @brief Visit Local Variable Definition (矢量 OR 标量)
 * @details: 
 *      varDef: lValue (ASSIGN initValue)?;
 *          1. lValue: ID (LBRACKET exp RBRACKET)*;
 *          2. initValue: exp | LBRACE (initValue (COMMA initValue)*)? RBRACE;
 * @note: 
 *      1. Constant: 必须使用常量初始化, 不能使用变量初始化
 *      2. Array: 
 *          2.1 必须使用常量初始化维度, 不能使用变量初始化
 *          2.2 既可以使用变量初始化数组元素, 也可以使用常量初始化数组元素
 */
ir::Value* SysYIRGenerator::visitVarDef_beta(SysYParser::VarDefContext* ctx,
                                             ir::Type* btype,
                                             bool is_const) {
    auto name = ctx->lValue()->ID()->getText();

    // 获得数组的各个维度 (常量)
    std::vector<ir::Value*> dims;
    int capacity = 1;
    for (auto dim : ctx->lValue()->exp()) {
        ir::Value* tmp = any_cast_Value(visit(dim));
        if (auto ctmp = dyn_cast<ir::Constant>(tmp)) {
            if (ctmp->is_float()) {
                capacity *= (int)(ctmp->f64());
                tmp = dyn_cast<ir::Value>(ir::Constant::gen_i32(ctmp->f64()));
            } else {
                capacity *= ctmp->i32();
            }
        } else {
            assert(false && "dimension must be a constant");
        }
        dims.push_back(tmp);
    }
    int dimensions = dims.size();
    bool isArray = dims.size() > 0;

    ir::Value* res = nullptr;
    ir::Value* init = nullptr;
    ir::Constant* cinit = nullptr;
    std::vector<ir::Value*> Arrayinit;

    //! get initial value and perform type conversion
    if (ctx->ASSIGN()) {
        if (isArray) {  //! 1. array initial value
            for (int i = 0; i < capacity; i++) {
                Arrayinit.push_back(nullptr);
            }

            _d = 0; _n = 0;
            _path.clear(); _path = std::vector<int>(dims.size(), 0);
            _current_type = btype; _is_alloca = true;

            // 将数组元素的初始化值存储在Arrayinit中
            for (auto expr : ctx->initValue()->initValue()) {
                visitInitValue_beta(expr, capacity, dims, Arrayinit);
            }
        } else {  //! 2. scalar initial value
            init = any_cast_Value(visit(ctx->initValue()->exp()));
            if (ir::isa<ir::Constant>(init)) {  // 2.1 常量
                cinit = dyn_cast<ir::Constant>(init);
                if (btype->is_i32() && cinit->is_float()) {
                    cinit = ir::Constant::gen_i32(cinit->f64());
                    init = dyn_cast<ir::Value>(cinit);
                } else if (btype->is_float() && cinit->is_i32()) {
                    cinit = ir::Constant::gen_f64(cinit->i32());
                    init = dyn_cast<ir::Value>(cinit);
                }
            } else {  // 2.2 变量
                if (btype->is_i32() && init->is_float()) {
                    init = _builder.create_ftosi(ir::Type::i32_type(), init, _builder.getvarname());
                } else if (btype->is_float() && init->is_i32()) {
                    init = _builder.create_sitof(ir::Type::float_type(), init, _builder.getvarname());
                }
            }
        }
    }

    //! deal with assignment (左值)
    if (is_const) {  //! 1. const
        if (!isArray) {  //! 1.1 scalar
            if (ctx->ASSIGN()) {
                if (cinit) {
                    _tables.insert(name, cinit);
                    res = cinit;
                } else {
                    std::cerr << "can't use variable initialize const" << std::endl;
                    exit(EXIT_FAILURE);
                }
            } else {
                std::cerr << "const without initialization!" << std::endl;
                exit(EXIT_FAILURE);
            }
        } else {  //! 1.2 array
            auto alloca_ptr = _builder.create_alloca(btype, dims, _builder.getvarname(), is_const);
            _tables.insert(name, alloca_ptr);
            ir::Value* element_ptr = dyn_cast<ir::Value>(alloca_ptr);
            for (int cur = 1; cur <= dimensions; cur++) {
                element_ptr = _builder.create_getelementptr(btype, element_ptr, 
                                                            ir::Constant::gen_i32(0), cur, 
                                                            dims, _builder.getvarname(), 1);
            }
            int cnt = 0;
            for (int i = 0; i < Arrayinit.size(); i++) {
                if (Arrayinit[i]) {
                    if (i != 0) {
                        element_ptr = _builder.create_getelementptr(btype, element_ptr, 
                                                                    ir::Constant::gen_i32(cnt), i, 
                                                                    dims, _builder.getvarname(), 2);
                        cnt = 0;
                    }
                    auto store = _builder.create_store(Arrayinit[i], element_ptr, "store");
                }
                cnt++;
            }
        }
    } else {  //! 2. variable
        //! create alloca inst
        auto alloca_ptr = _builder.create_alloca(btype, dims, _builder.getvarname());
        _tables.insert(name, alloca_ptr);
        
        //! create store inst
        if (!isArray) {  //! 2.1 scalar
            if(ctx->ASSIGN()){
                auto store = _builder.create_store(init, alloca_ptr, "store");
            }
        } else {  //! 2.2 array
            if(ctx->ASSIGN()) {
                ir::Value* element_ptr = dyn_cast<ir::Value>(alloca_ptr);
                for (int cur = 1; cur <= dimensions; cur++) {
                    element_ptr = _builder.create_getelementptr(btype, element_ptr, 
                                                                ir::Constant::gen_i32(0), cur, 
                                                                dims, _builder.getvarname(), 1);
                }
                int cnt = 0;
                for (int i = 0; i < Arrayinit.size(); i++) {
                    if (Arrayinit[i]) {
                        if (i != 0) {
                            element_ptr = _builder.create_getelementptr(btype, element_ptr, 
                                                                        ir::Constant::gen_i32(cnt), i, 
                                                                        dims, _builder.getvarname(), 2);
                            cnt = 0;
                        }
                        auto store = _builder.create_store(Arrayinit[i], element_ptr, "store");
                    }
                    cnt++;
                }
            }
        }

        res = alloca_ptr;
    }
    return dyn_cast_Value(res);
}

/*
 * @brief Visit Global Variable Definition (矢量 OR 标量)
 * @details: 
 *      varDef: lValue (ASSIGN initValue)?;
 *          1. lValue: ID (LBRACKET exp RBRACKET)*;
 *          2. initValue: exp | LBRACE (initValue (COMMA initValue)*)? RBRACE;
 * @note: 
 *      1. Variable/Constant: 必须使用常值表达式进行初始化, 不能使用变量初始化
 */
ir::Value* SysYIRGenerator::visitVarDef_global(SysYParser::VarDefContext* ctx,
                                               ir::Type* btype, bool is_const) {
    auto name = ctx->lValue()->ID()->getText();

    // 获得数组的各个维度 (常量)
    std::vector<ir::Value*> dims;
    int capacity = 1;
    for (auto dim : ctx->lValue()->exp()) {
        ir::Value* tmp = any_cast_Value(visit(dim));
        if (auto ctmp = dyn_cast<ir::Constant>(tmp)) {
            if (ctmp->is_float()) {
                capacity *= (int)(ctmp->f64());
                tmp = dyn_cast<ir::Value>(ir::Constant::gen_i32(ctmp->f64()));
            } else {
                capacity *= ctmp->i32();
            }
        } else {
            assert(false && "dimension must be a constant");
        }
        dims.push_back(tmp);
    }
    bool isArray = dims.size() > 0;

    ir::Value* res = nullptr;
    std::vector<ir::Value*> init;
    ir::Constant* cinit;

    //! get initial value and perform type conversion
    if (ctx->ASSIGN()) {
        if (isArray) {  //! 1. array initial value
            for (int i = 0; i < capacity; i++) {
                init.push_back(ir::Constant::gen_i32(0));
            }

            _d = 0; _n = 0;
            _path.clear(); _path = std::vector<int>(dims.size(), 0);
            _current_type = btype; _is_alloca = false;
            
            // 将数组元素的初始化值存储在init中
            for (auto expr : ctx->initValue()->initValue()) {
                visitInitValue_beta(expr, capacity, dims, init);
            }
        } else {  //! 2. scalar initial value
            ir::Value* tmp = any_cast_Value(visit(ctx->initValue()->exp()));
            assert(ir::isa<ir::Constant>(tmp) && "Global must be initialized by constant");
            
            cinit = dyn_cast<ir::Constant>(tmp);
            if (btype->is_i32() && cinit->is_float()) {
                cinit = ir::Constant::gen_i32(cinit->f64(), "@" + name);
            } else if (btype->is_float() && cinit->is_i32()) {
                cinit = ir::Constant::gen_f64(cinit->i32(), "@" + name);
            } else if (cinit->is_float()) {
                cinit = ir::Constant::gen_f64(cinit->f64(), "@" + name);
            } else if (cinit->is_i32()) {
                cinit = ir::Constant::gen_i32(cinit->i32(), "@" + name);
            } else {
                assert(false && "Invalid type");
            }
            init.push_back(cinit);
        }
    } else {  // note: default init is zero
        if (isArray) {  //! 1. array initial value
            // pass (不进行任何操作)
        } else {  //! 2. scalar initial value
            if (btype->is_i32()) {
                cinit = ir::Constant::gen_i32(0, "@" + name);
            } else if (btype->is_float()) {
                cinit = ir::Constant::gen_f64(0.0, "@" + name);
            } else {
                assert(false && "Invalid type");
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
            auto gv = ir::GlobalVariable::gen(btype, init, dims, _module, "@" + name, true);
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
    return dyn_cast_Value(res);
}


/*
 * @brief Visit Global Variable Declaration (全局变量定义 && 声明)
 * @details
 *      decl: CONST? btype varDef (COMMA varDef)* SEMICOLON;
 */
ir::Value* SysYIRGenerator::visitDeclGlobal(SysYParser::DeclContext* ctx) {
    auto btype = any_cast_Type(visit(ctx->btype()));
    bool is_const = ctx->CONST();
    ir::Value* res = nullptr;
    for (auto varDef : ctx->varDef()) {
        res = visitVarDef_global(varDef, btype, is_const);
    }

    return dyn_cast_Value(res);
}

/*
 * @brief visit initvalue
 *      varDef: lValue (ASSIGN initValue)?;
 *      initValue: exp | LBRACE (initValue (COMMA initValue)*)? RBRACE;
 */
void SysYIRGenerator::visitInitValue_beta(SysYParser::InitValueContext *ctx, 
                                              const int capacity, const std::vector<ir::Value*> dims, 
                                              std::vector<ir::Value*>& init) {
    if (ctx->exp()) {
        auto value = any_cast_Value(visit(ctx->exp()));

        //! 类型转换 (匹配左值与右值的数据类型)
        if (auto cvalue = dyn_cast<ir::Constant>(value)) {  //! 1. 常量
            if (_current_type->is_i32() && cvalue->is_float()) {
                value = ir::Constant::gen_i32(cvalue->f64());
            } else if (_current_type->is_float() && cvalue->is_i32()) {
                value = ir::Constant::gen_f64(cvalue->i32());
            }
        } else {  //! 2. 变量
            if (_current_type->is_i32() && value->is_float()) {
                value = _builder.create_ftosi(ir::Type::i32_type(), value, _builder.getvarname());
            } else if (_current_type->is_float() && value->is_i32()) {
                value = _builder.create_sitof(ir::Type::float_type(), value, _builder.getvarname());
            }
        }

        //! 获取当前数组元素的位置
        while (_d < dims.size() - 1) {
            _path[_d++] = _n;
            _n = 0;
        }
        std::vector<ir::Value*> indices;  // 大小为数组维度 (存储当前visit的元素的下标)
        for (int i = 0; i < dims.size() - 1; i++) {
            indices.push_back(ir::Constant::gen_i32(_path[i]));
        }
        indices.push_back(ir::Constant::gen_i32(_n));

        //! 将特定位置的数组元素存入init数组中
        int factor = 1, offset = 0;
        for (int i = indices.size() - 1; i >= 0; i--) {
            offset += factor * dyn_cast<ir::Constant>(indices[i])->i32();
            factor *= dyn_cast<ir::Constant>(dims[i])->i32();
        }
        if (auto cvalue = dyn_cast<ir::Constant>(value)) {  // 1. 常值 (global OR local)
            init[offset] = value;
        } else {  // 2. 变量 (just for local)
            if (_is_alloca) {
                init[offset] = value;
            }else {
                assert(false && "global variable must be initialized by constant");
            }
        }
    } else {
        int cur_d = _d, cur_n = _n;
        for (auto expr : ctx->initValue()) {
            visitInitValue_beta(expr, capacity, dims, init);
        }
        _d = cur_d, _n = cur_n;
    }

    // goto next element
    _n++;
    while (_d >= 0 && _n >= dyn_cast<ir::Constant>(dims[_d])->i32()) {
        _n = _path[--_d] + 1;
    }
}

}  // namespace sysy