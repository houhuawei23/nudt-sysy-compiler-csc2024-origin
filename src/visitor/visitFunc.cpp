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
    return ir::Type::TypeInt32();
  } else if (ctx->FLOAT()) {
    return ir::Type::TypeFloat32();
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
  const auto func_name = ctx->ID()->getText();
  std::vector<ir::Type*> param_types;

  if (ctx->funcFParams()) {
    auto params = ctx->funcFParams()->funcFParam();

    for (auto param : params) {
      bool isArray = not param->LBRACKET().empty();

      auto base_type = any_cast_Type(visit(param->btype()));

      if (!isArray) {
        param_types.push_back(base_type);
      } else {
        std::vector<size_t> dims;
        for (auto expr : param->exp()) {
          auto value = any_cast_Value(visit(expr));
          if (auto cvalue = dyn_cast<ir::Constant>(value)) {
            if (cvalue->isFloatPoint())
              dims.push_back((int)cvalue->f32());
            else
              dims.push_back(cvalue->i32());
          } else {
            assert(false && "function parameter must be constant");
          }
        }

        if (dims.size() != 0)
          base_type = ir::Type::TypeArray(base_type, dims);
        param_types.push_back(ir::Type::TypePointer(base_type));
      }
    }
  }

  const auto ret_type = any_cast_Type(visit(ctx->funcType()));
  const auto TypeFunction = ir::Type::TypeFunction(ret_type, param_types);
  auto func = mModule->addFunction(TypeFunction, func_name);

  return func;
}

/*
 * @brief: visit function define
 * @details:
 *      funcDef: funcType ID LPAREN funcFParams? RPAREN blockStmt;
 *      funcFParams: funcFParam (COMMA funcFParam)*;
 *      funcFParam: btype ID (LBRACKET RBRACKET (LBRACKET exp RBRACKET)*)?;
 * entry -> next -> other -> .... -> exit
 * entry: allocas, br
 * next: retval, params, br
 * other: blockStmt init block
 * exit: load retval, ret
 */
std::any SysYIRGenerator::visitFuncDef(SysYParser::FuncDefContext* ctx) {
  const auto func_name = ctx->ID()->getText();
  auto func = mModule->findFunction(func_name);
  if (not func)
    func = create_func(ctx);

  if (ctx->blockStmt()) {
    // create and enter function scope
    // it will be automatically destroyed when return from this visitFfunc
    ir::SymbolTable::FunctionScope scope(mTables);

    func->newEntry()->addComment("entry");
    func->newExit()->addComment("exit");

    const auto next = func->newBlock();
    next->addComment("next");

    func->entry()->emplace_back_inst(
        mBuilder.makeIdenticalInst<ir::BranchInst>(next));

    const auto retType = func->retType();
    // next block
    {
      mBuilder.set_pos(next);
      // create return value alloca
      if (not retType->isVoid()) {
        auto ret_value_ptr = mBuilder.makeAlloca(retType);
        ret_value_ptr->addComment("retval");
        switch (retType->btype()) {
          case ir::BasicTypeRank::INT32:
            mBuilder.makeInst<ir::StoreInst>(ir::Constant::gen_i32(0),
                                             ret_value_ptr);
            break;
          case ir::BasicTypeRank::FLOAT:
            mBuilder.makeInst<ir::StoreInst>(ir::Constant::gen_f32(0.0),
                                             ret_value_ptr);
            break;
          default:
            assert(false && "not valid type");
        }
        func->setRetValueAddr(ret_value_ptr);
      }
      // create params and alloca
      if (ctx->funcFParams()) {
        // first init args
        for (auto pram : ctx->funcFParams()->funcFParam()) {
          const auto arg_name = pram->ID()->getText();

          bool isArray = not pram->LBRACKET().empty();
          auto arg_type = any_cast_Type(visit(pram->btype()));
          if (isArray) {
            std::vector<size_t> dims;
            for (auto expr : pram->exp()) {
              auto value = any_cast_Value(visit(expr));
              if (auto cvalue = dyn_cast<ir::Constant>(value)) {
                if (cvalue->isFloatPoint())
                  dims.push_back((int)cvalue->f32());
                else
                  dims.push_back(cvalue->i32());
              } else {
                assert(false && "function parameter must be constant");
              }
            }
            if (dims.size() != 0)
              arg_type = ir::Type::TypeArray(arg_type, dims);
            arg_type = ir::Type::TypePointer(arg_type);
          }

          auto arg = func->new_arg(arg_type);
        }

        // allca all params and store
        int idx = 0;
        for (auto pram : ctx->funcFParams()->funcFParam()) {
          const auto arg_name = pram->ID()->getText();

          bool isArray = not pram->LBRACKET().empty();
          auto arg_type = any_cast_Type(visit(pram->btype()));
          if (isArray) {
            std::vector<size_t> dims;
            for (auto expr : pram->exp()) {
              auto value = any_cast_Value(visit(expr));
              if (auto cvalue = dyn_cast<ir::Constant>(value)) {
                if (cvalue->isFloatPoint())
                  dims.push_back((int)cvalue->f32());
                else
                  dims.push_back(cvalue->i32());
              } else {
                assert(false && "function parameter must be constant");
              }
            }
            if (dims.size() != 0)
              arg_type = ir::Type::TypeArray(arg_type, dims);
            arg_type = ir::Type::TypePointer(arg_type);
          }

          auto alloca_ptr = mBuilder.makeAlloca(arg_type);
          alloca_ptr->addComment(arg_name);

          auto store =
              mBuilder.makeInst<ir::StoreInst>(func->arg_i(idx), alloca_ptr);

          mTables.insert(arg_name, alloca_ptr);
          idx++;
        }
      }
    }

    const auto other = func->newBlock();
    other->addComment("other");
    next->emplace_back_inst(mBuilder.makeIdenticalInst<ir::BranchInst>(other));
    /* other is the init block of blockStmt */
    {
      mBuilder.set_pos(other);
      visitBlockStmt(ctx->blockStmt());
      if (not mBuilder.curBlock()->isTerminal()) {
        mBuilder.makeInst<ir::BranchInst>(func->exit());
      }
    }

    /* exit is the last block of function */
    {
      mBuilder.set_pos(func->exit());

      if (not retType->isVoid()) {
        mBuilder.makeInst<ir::ReturnInst>(mBuilder.makeLoad(func->retValPtr()));
      } else {
        mBuilder.makeInst<ir::ReturnInst>(nullptr);
      }
    }
  }

  mBuilder.reset();
  return dyn_cast_Value(func);
}

}  // namespace sysy
