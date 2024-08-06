#include "visitor/visitor.hpp"

using namespace ir;
namespace sysy {
/*
 * @brief Visit Variable Type (变量类型)
 * @details
 *      btype: INT | FLOAT;
 */
std::any SysYIRGenerator::visitBtype(SysYParser::BtypeContext* ctx) {
  if (ctx->INT()) {
    return Type::TypeInt32();
  } else if (ctx->FLOAT()) {
    return Type::TypeFloat32();
  }
  return nullptr;
}

/*
 * @brief Visit Variable Declaration (变量定义 && 声明)
 * @details
 *      Global OR Local (全局 OR 局部)
 */
std::any SysYIRGenerator::visitDecl(SysYParser::DeclContext* ctx) {
  auto btype = any_cast_Type(visit(ctx->btype()));
  bool is_const = ctx->CONST();
  // std::cerr << ctx->getText() << std::endl;
  for (auto varDef : ctx->varDef()) {
    visitVarDef(varDef, btype, is_const);
  }

  return std::any();
}

// Value* expandConstant(Instruction* val) {
//   return val->getConstantRepl
// }


Value* SysYIRGenerator::visitVarDef(SysYParser::VarDefContext* ctx, Type* btype, bool is_const) {
  // 获得数组的各个维度 (常量)
  std::vector<size_t> dims;
  size_t capacity = 1;
  // std::cerr << "visitVarDef: " << ctx->getText() << std::endl;
  for (auto dimCtx : ctx->lValue()->exp()) {
    auto dim = any_cast_Value(visit(dimCtx));
    if(auto instdim = dim->dynCast<Instruction>())
      dim = instdim->getConstantRepl(true);
    assert(dim->isa<ConstantValue>() && "dimension must be a constant");
    auto cdim = dim->dynCast<ConstantValue>();
    capacity *= cdim->i32();
    dims.push_back(cdim->i32());
  }
  bool isArray = dims.size() > 0;

  // std::cerr << ctx->getText() << std::endl;
  if (mTables.isModuleScope()) {
    if (isArray)
      return visitGlobalArray(ctx, btype, is_const, dims, capacity);
    else
      return visitGlobalScalar(ctx, btype, is_const);
  } else {
    if (isArray)
      return visitLocalArray(ctx, btype, is_const, dims, capacity);
    else
      return visitLocalScalar(ctx, btype, is_const);
  }
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
Value* SysYIRGenerator::visitGlobalArray(SysYParser::VarDefContext* ctx,
                                         Type* btype,
                                         bool is_const,
                                         std::vector<size_t> dims,
                                         int capacity) {
  const auto name = ctx->lValue()->ID()->getText();

  std::vector<Value*> Arrayinit;
  bool is_init = false;
  for (int i = 0; i < capacity; i++) {
    if (btype->isFloatPoint()) {
      Arrayinit.push_back(ConstantFloating::gen_f32(0.0));
    } else if (btype->isInt32()) {
      Arrayinit.push_back(ConstantInteger::gen_i32(0));
    } else {
      assert(false && "Invalid type.");
    }
  }

  //! get initial value (将数组元素的初始化值存储在Arrayinit中)
  if (ctx->ASSIGN()) {
    _d = 0; _n = 0; _path.clear();
    _path = std::vector<size_t>(dims.size(), 0);
    _current_type = btype; _is_alloca = true;
    for (auto expr : ctx->initValue()->initValue()) {
      is_init |= visitInitValue_Array(expr, capacity, dims, Arrayinit);
    }
  }

  //! generate global variable and assign
  auto global_var = GlobalVariable::gen(btype, Arrayinit, mModule, name, is_const, dims, is_init, capacity);
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
Value* SysYIRGenerator::visitGlobalScalar(SysYParser::VarDefContext* ctx,
                                          Type* btype,
                                          bool is_const) {
  const auto name = ctx->lValue()->ID()->getText();

  Value* init = nullptr;
  bool is_init = false;
  if (btype->isFloatPoint()) {
    init = ConstantFloating::gen_f32(0.0);
  } else if (btype->isInt32()) {
    init = ConstantInteger::gen_i32(0);
  } else {
    assert(false && "invalid type");
  }
  if (ctx->ASSIGN()) {
    is_init = true;
    init = any_cast_Value(visit(ctx->initValue()->exp()));
    if(auto initInst = init->dynCast<Instruction>()) 
      init = initInst->getConstantRepl(true);
    assert(init->isa<ConstantValue>() && "global must be initialized by constant");
    init = mBuilder.castConstantType(init, btype);
  }

  //! generate global variable and assign
  auto global_var = GlobalVariable::gen(btype, {init}, mModule, name, is_const, {}, is_init);
  mTables.insert(name, global_var);
  mModule->addGlobalVar(name, global_var);

  return dyn_cast_Value(global_var);
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
Value* SysYIRGenerator::visitLocalArray(SysYParser::VarDefContext* ctx,
                                        Type* btype,
                                        bool is_const,
                                        std::vector<size_t> dims,
                                        int capacity) {
  const auto name = ctx->lValue()->ID()->getText();
  int dimensions = dims.size();
  std::vector<size_t> cur_dims(dims);

  std::vector<Value*> Arrayinit;

  //! alloca
  auto alloca_ptr = mBuilder.makeAlloca(btype, is_const, dims, name, capacity);
  mTables.insert(name, alloca_ptr);

  //! get initial value (将数组元素的初始化值存储在Arrayinit中)
  if (ctx->ASSIGN()) {
    for (int i = 0; i < capacity; i++)
      Arrayinit.push_back(nullptr);
    auto tmp = mBuilder.makeInst<BitCastInst>(alloca_ptr->type(), alloca_ptr);
    mBuilder.makeInst<MemsetInst>(tmp->type(), tmp);

    // auto ptr = mBuilder.makeInst<BitCastInst>(Type::TypeInt8(), alloca_ptr);
    // // mBuilder.makeInst<MemsetInst>(tmp->type(), tmp);
    // const auto len = alloca_ptr->type()->as<PointerType>()->baseType()->size();
    // // mBuilder.makeInst<MemsetInstBeta>(ptr, ConstantInteger::gen_i8(0),
    // ConstantInteger::gen_i32(len));

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
  bool isAssign = false;
  for (int i = 0; i < Arrayinit.size(); i++) {
    if (Arrayinit[i] != nullptr) {
      isAssign = true;
      break;
    }
  }

  if (!isAssign)
    return dyn_cast_Value(alloca_ptr);
  Value* element_ptr = dyn_cast<Value>(alloca_ptr);
  for (int cur = 1; cur <= dimensions; cur++) {
    dims.erase(dims.begin());
    element_ptr =
      mBuilder.makeGetElementPtr(btype, element_ptr, ConstantInteger::gen_i32(0), dims, cur_dims);
    cur_dims.erase(cur_dims.begin());
  }

  int cnt = 0;
  for (int i = 0; i < Arrayinit.size(); i++) {
    if (Arrayinit[i] != nullptr) {
      element_ptr = mBuilder.makeGetElementPtr(btype, element_ptr, ConstantInteger::gen_i32(cnt));
      mBuilder.makeInst<StoreInst>(Arrayinit[i], element_ptr);
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
Value* SysYIRGenerator::visitLocalScalar(SysYParser::VarDefContext* ctx,
                                         Type* btype,
                                         bool is_const) {
  const auto name = ctx->lValue()->ID()->getText();

  if (is_const) {  //! const qulifier
    if (!ctx->ASSIGN())
      assert(false && "const without initialization");
    auto init = any_cast_Value(visit(ctx->initValue()->exp()));

    assert(init->isa<ConstantValue>() && "const must be initialized by constant");

    // init is Constant
    init = mBuilder.castConstantType(init, btype);
    mTables.insert(name, init);

    return init;
  } else {  //! not const qulifier
    auto alloca_ptr = mBuilder.makeAlloca(btype, is_const)->setComment(name);
    mTables.insert(name, alloca_ptr);

    if (not ctx->ASSIGN())
      return alloca_ptr;

    // has init
    auto init = any_cast_Value(visit(ctx->initValue()->exp()));
    if (init->isa<ConstantValue>())
      init = mBuilder.castConstantType(init, btype);
    else
      init = mBuilder.makeTypeCast(init, btype);

    mBuilder.makeInst<StoreInst>(init, alloca_ptr);

    return alloca_ptr;
  }
}

/*
 * @brief: visit array initvalue
 * @details:
 *      varDef: lValue (ASSIGN initValue)?;
 *      initValue: exp | LBRACE (initValue (COMMA initValue)*)? RBRACE;
 */
bool SysYIRGenerator::visitInitValue_Array(SysYParser::InitValueContext* ctx,
                                           const int capacity,
                                           const std::vector<size_t> dims,
                                           std::vector<Value*>& init) {
  bool res = false;
  if (ctx->exp()) {
    auto value = any_cast_Value(visit(ctx->exp()));

    //! 类型转换 (匹配左值与右值的数据类型)
    if (value->isa<ConstantValue>())
      value = mBuilder.castConstantType(value, _current_type);
    else
      value = mBuilder.makeTypeCast(value, _current_type);

    //! 获取当前数组元素的位置
    while (_d < dims.size() - 1) {
      _path[_d++] = _n;
      _n = 0;
    }
    std::vector<Value*> indices;  // 大小为数组维度 (存储当前visit的元素的下标)
    for (int i = 0; i < dims.size() - 1; i++) {
      indices.push_back(ConstantInteger::gen_i32(_path[i]));
    }
    indices.push_back(ConstantInteger::gen_i32(_n));

    //! 将特定位置的数组元素存入init数组中
    int factor = 1, offset = 0;
    for (int i = indices.size() - 1; i >= 0; i--) {
      offset += factor * indices[i]->dynCast<ConstantInteger>()->getVal();
      factor *= dims[i];
    }
    if (auto cvalue = value->dynCast<ConstantInteger>()) {  // 1. 常值 (global OR local)
      res = true;
      init[offset] = value;
    } else {  // 2. 变量 (just for local)
      res = true;
      if (_is_alloca) {
        init[offset] = value;
      } else {
        assert(false && "global variable must be initialized by constant");
      }
    }
  } else {
    int cur_d = _d, cur_n = _n;
    for (auto expr : ctx->initValue()) {
      res |= visitInitValue_Array(expr, capacity, dims, init);
    }
    _d = cur_d, _n = cur_n;
  }

  // goto next element
  _n++;
  while (_d >= 0 && _n >= dims[_d]) {
    _n = _path[--_d] + 1;
  }
  return res;
}

}  // namespace sysy