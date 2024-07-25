#include "visitor/visitor.hpp"

namespace sysy {
/*
 * @brief Visit Variable Type (变量类型)
 * @details
 *      btype: INT | FLOAT;
 */
std::any SysYIRGenerator::visitBtype(SysYParser::BtypeContext* ctx) {
  if (ctx->INT()) {
    return ir::Type::TypeInt32();
  } else if (ctx->FLOAT()) {
    return ir::Type::TypeFloat32();
  }
  return nullptr;
}

/*
 * @brief Visit Variable Declaration (变量定义 && 声明)
 * @details
 *      Global OR Local (全局 OR 局部)
 */
std::any SysYIRGenerator::visitDecl(SysYParser::DeclContext* ctx) {
  if (mTables.isModuleScope()) {
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
    res = visitVarDef_local(varDef, btype, is_const);
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

ir::Value* SysYIRGenerator::visitVarDef_global(SysYParser::VarDefContext* ctx, ir::Type* btype, bool is_const) {
  const auto name = ctx->lValue()->ID()->getText();

  // 获得数组的各个维度 (常量)
  std::vector<size_t> dims;
  int capacity = 1;
  for (auto dim : ctx->lValue()->exp()) {
    ir::Value* tmp = any_cast_Value(visit(dim));
    if (auto ctmp = dyn_cast<ir::Constant>(tmp)) {
      if (ctmp->isFloatPoint()) {
        capacity *= (int)(ctmp->f32());
        dims.push_back((int)ctmp->f32());
      } else {
        capacity *= ctmp->i32();
        dims.push_back(ctmp->i32());
      }
    } else {
      assert(false && "dimension must be a constant");
    }
  }
  bool isArray = dims.size() > 0;

  if (isArray) return visitArray_global(ctx, btype, is_const, dims, capacity);
  else return visitScalar_global(ctx, btype, is_const);
}
/*
 * @brief: visit global array
 * @details:
 *      varDef: lValue (ASSIGN initValue)?;
 *      lValue: ID (LBRACKET exp RBRACKET)*;
 *      initValue: exp | LBRACE (initValue (COMMA initValue)*)? RBRACE;
 * @note: global variable
 *      1. const
 *      2. variable
 */
ir::Value* SysYIRGenerator::visitArray_global(SysYParser::VarDefContext* ctx, ir::Type* btype, bool is_const,
                                              std::vector<size_t> dims, int capacity) {
  const auto name = ctx->lValue()->ID()->getText();
  int dimensions = dims.size();

  std::vector<ir::Value*> Arrayinit;
  for (int i = 0; i < capacity; i++) {
    if (btype->isFloatPoint())
      Arrayinit.push_back(ir::Constant::gen_f32(0.0));
    else if (btype->isInt32())
      Arrayinit.push_back(ir::Constant::gen_i32(0));
    else
      assert(false && "Invalid type.");
  }

  //! get initial value (将数组元素的初始化值存储在Arrayinit中)
  if (ctx->ASSIGN()) {
    _d = 0;
    _n = 0;
    _path.clear();
    _path = std::vector<size_t>(dims.size(), 0);
    _current_type = btype;
    _is_alloca = true;
    for (auto expr : ctx->initValue()->initValue()) {
      visitInitValue_Array(expr, capacity, dims, Arrayinit);
    }
  }

  //! generate global variable and assign
  auto global_var = ir::GlobalVariable::gen(btype, Arrayinit, mModule, name, is_const, dims);
  mTables.insert(name, global_var);
  mModule->addGlobalVar(name, global_var);

  return dyn_cast_Value(global_var);
}
/*
 * @brief: visit global scalar
 * @details:
 *      varDef: lValue (ASSIGN initValue)?;
 *      lValue: ID (LBRACKET exp RBRACKET)*;
 *      initValue: exp | LBRACE (initValue (COMMA initValue)*)? RBRACE;
 * @note: global variable
 *      1. const
 *      2. variable
 */
ir::Value* SysYIRGenerator::visitScalar_global(SysYParser::VarDefContext* ctx,
                                               ir::Type* btype,
                                               bool is_const) {
  const auto name = ctx->lValue()->ID()->getText();

  ir::Value* init = nullptr;
  if (btype->isFloatPoint())
    init = ir::Constant::gen_f32(0.0);
  else if (btype->isInt32())
    init = ir::Constant::gen_i32(0);
  else
    assert(false && "invalid type");

  if (ctx->ASSIGN()) {
    init = any_cast_Value(visit(ctx->initValue()->exp()));
    assert(ir::isa<ir::Constant>(init) &&
           "global must be initialized by constant");

    ir::Constant* tmp = dyn_cast<ir::Constant>(init);
    if (btype->isInt32() && tmp->isFloatPoint()) {
      init = ir::Constant::gen_i32(tmp->f32());
    } else if (btype->isFloatPoint() && tmp->isInt32()) {
      init = ir::Constant::gen_f32(tmp->i32());
    }
  }

  //! generate global variable and assign
  auto global_var =
      ir::GlobalVariable::gen(btype, {init}, mModule, name, is_const);
  mTables.insert(name, global_var);
  mModule->addGlobalVar(name, global_var);

  return dyn_cast_Value(global_var);
}

ir::Value* SysYIRGenerator::visitVarDef_local(SysYParser::VarDefContext* ctx,
                                              ir::Type* btype,
                                              bool is_const) {
  // 获得数组的各个维度 (常量)
  std::vector<size_t> dims;
  int capacity = 1;
  for (auto dim : ctx->lValue()->exp()) {
    ir::Value* tmp = any_cast_Value(visit(dim));
    if (auto ctmp = dyn_cast<ir::Constant>(tmp)) {
      if (ctmp->isFloatPoint()) {
        capacity *= (int)(ctmp->f32());
        dims.push_back((int)ctmp->f32());
      } else if (ctmp->isInt32()) {
        capacity *= ctmp->i32();
        dims.push_back(ctmp->i32());
      } else {
        assert(false && "Invalid type.");
      }
    } else {
      assert(false && "dimension must be a constant");
    }
  }
  bool isArray = dims.size() > 0;

  if (isArray)
    return visitArray_local(ctx, btype, is_const, dims, capacity);
  else
    return visitScalar_local(ctx, btype, is_const);
}
/*
 * @brief: visit local array
 * @details:
 *      varDef: lValue (ASSIGN initValue)?;
 *      lValue: ID (LBRACKET exp RBRACKET)*;
 *      initValue: exp | LBRACE (initValue (COMMA initValue)*)? RBRACE;
 * @note: alloca
 *      1. const
 *      2. variable
 */
ir::Value* SysYIRGenerator::visitArray_local(SysYParser::VarDefContext* ctx, ir::Type* btype,
                                             bool is_const, std::vector<size_t> dims, int capacity) {
  const auto name = ctx->lValue()->ID()->getText();
  int dimensions = dims.size();
  std::vector<size_t> cur_dims(dims);

  std::vector<ir::Value*> Arrayinit;

  //! alloca
  auto alloca_ptr = mBuilder.makeAlloca(btype, is_const, dims, name, capacity);
  mTables.insert(name, alloca_ptr);

  //! get initial value (将数组元素的初始化值存储在Arrayinit中)
  if (ctx->ASSIGN()) {
    for (int i = 0; i < capacity; i++)
      Arrayinit.push_back(nullptr);
    auto tmp =
        mBuilder.makeInst<ir::BitCastInst>(alloca_ptr->type(), alloca_ptr);
    mBuilder.makeInst<ir::MemsetInst>(tmp->type(), tmp);

    _d = 0;
    _n = 0;
    _path.clear();
    _path = std::vector<size_t>(dims.size(), 0);
    _current_type = btype;
    _is_alloca = true;
    for (auto expr : ctx->initValue()->initValue()) {
      visitInitValue_Array(expr, capacity, dims, Arrayinit);
    }
  }

  //! assign
  ir::Value* element_ptr = dyn_cast<ir::Value>(alloca_ptr);
  for (int cur = 1; cur <= dimensions; cur++) {
    dims.erase(dims.begin());
    element_ptr = mBuilder.makeGetElementPtr(btype, element_ptr, ir::Constant::gen_i32(0), dims, cur_dims);
    cur_dims.erase(cur_dims.begin());
  }

  int cnt = 0;
  for (int i = 0; i < Arrayinit.size(); i++) {
    if (Arrayinit[i] != nullptr) {
      element_ptr = mBuilder.makeGetElementPtr(btype, element_ptr,
                                               ir::Constant::gen_i32(cnt));
      // mBuilder.create_store(Arrayinit[i], element_ptr);
      mBuilder.makeInst<ir::StoreInst>(Arrayinit[i], element_ptr);
      cnt = 0;
    }
    cnt++;
  }

  return dyn_cast_Value(alloca_ptr);
}
/*
 * @brief: visit local scalar
 * @details:
 *      varDef: lValue (ASSIGN initValue)?;
 *      lValue: ID (LBRACKET exp RBRACKET)*;
 *      initValue: exp | LBRACE (initValue (COMMA initValue)*)? RBRACE;
 * @note:
 *      1. const     ignore
 *      2. variable  alloca
 */
ir::Value* SysYIRGenerator::visitScalar_local(SysYParser::VarDefContext* ctx, ir::Type* btype, bool is_const) {
  const auto name = ctx->lValue()->ID()->getText();
  ir::Value* init = nullptr;

  if (is_const) {  //! 常量
    if (!ctx->ASSIGN())
      assert(false && "const without initialization");
    init = any_cast_Value(visit(ctx->initValue()->exp()));
    if (ir::isa<ir::Constant>(init)) {  //! 1. 右值为常量
      auto cinit = dyn_cast<ir::Constant>(init);
      if (btype->isInt32() && cinit->isFloatPoint()) {
        cinit = ir::Constant::gen_i32(cinit->f32());
        init = dyn_cast<ir::Value>(cinit);
      } else if (btype->isFloatPoint() && cinit->isInt32()) {
        cinit = ir::Constant::gen_f32(cinit->i32());
        init = dyn_cast<ir::Value>(cinit);
      }
      mTables.insert(name, init);
    } else {  //! 2. 右值为变量
      assert(false && "const must be initialized by constant");
    }

    return init;
  } else {  //! 变量
    auto alloca_ptr = mBuilder.makeAlloca(btype, is_const);
    alloca_ptr->setComment(name);
    mTables.insert(name, alloca_ptr);

    if (ctx->ASSIGN()) {
      init = any_cast_Value(visit(ctx->initValue()->exp()));
      if (ir::isa<ir::Constant>(init)) {  //! 1. 右值为常量
        auto cinit = dyn_cast<ir::Constant>(init);
        if (btype->isInt32() && cinit->isFloatPoint()) {
          cinit = ir::Constant::gen_i32(cinit->f32());
          init = dyn_cast<ir::Value>(cinit);
        } else if (btype->isFloatPoint() && cinit->isInt32()) {
          cinit = ir::Constant::gen_f32(cinit->i32());
          init = dyn_cast<ir::Value>(cinit);
        }
      } else {  //! 2. 右值为变量
        if (btype->isInt32() && init->isFloatPoint()) {
          init = mBuilder.makeUnary(ir::vFPTOSI, init);
        } else if (btype->isFloatPoint() && init->isInt32()) {
          init = mBuilder.makeUnary(ir::vSITOFP, init);
        }
      }
      mBuilder.makeInst<ir::StoreInst>(init, alloca_ptr);
    }

    return dyn_cast_Value(alloca_ptr);
  }
}

/*
 * @brief visit array initvalue
 * @details:
 *      varDef: lValue (ASSIGN initValue)?;
 *      initValue: exp | LBRACE (initValue (COMMA initValue)*)? RBRACE;
 */
void SysYIRGenerator::visitInitValue_Array(SysYParser::InitValueContext* ctx, const int capacity,
                                           const std::vector<size_t> dims, std::vector<ir::Value*>& init) {
  if (ctx->exp()) {
    auto value = any_cast_Value(visit(ctx->exp()));

    //! 类型转换 (匹配左值与右值的数据类型)
    if (auto cvalue = dyn_cast<ir::Constant>(value)) {  //! 1. 常量
      if (_current_type->isInt32() && cvalue->isFloatPoint()) {
        value = ir::Constant::gen_i32(cvalue->f32());
      } else if (_current_type->isFloatPoint() && cvalue->isInt32()) {
        value = ir::Constant::gen_f32(cvalue->i32());
      }
    } else {  //! 2. 变量
      if (_current_type->isInt32() && value->isFloatPoint()) {
        value = mBuilder.makeUnary(ir::ValueId::vFPTOSI, value);
      } else if (_current_type->isFloatPoint() && value->isInt32()) {
        value = mBuilder.makeUnary(ir::ValueId::vSITOFP, value);
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
      factor *= dims[i];
    }
    if (auto cvalue = dyn_cast<ir::Constant>(value)) {  // 1. 常值 (global OR local)
      init[offset] = value;
    } else {  // 2. 变量 (just for local)
      if (_is_alloca) {
        init[offset] = value;
      } else {
        assert(false && "global variable must be initialized by constant");
      }
    }
  } else {
    int cur_d = _d, cur_n = _n;
    for (auto expr : ctx->initValue()) {
      visitInitValue_Array(expr, capacity, dims, init);
    }
    _d = cur_d, _n = cur_n;
  }

  // goto next element
  _n++;
  while (_d >= 0 && _n >= dims[_d]) {
    _n = _path[--_d] + 1;
  }
}

}  // namespace sysy