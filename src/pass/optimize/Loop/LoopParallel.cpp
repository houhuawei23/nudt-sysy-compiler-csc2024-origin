#include "pass/optimize/optimize.hpp"
#include "pass/optimize/Loop/LoopParallel.hpp"
#include "pass/analysis/ControlFlowGraph.hpp"
#include "pass/optimize/Loop/LoopBodyExtract.hpp"
// #include "libgen.h"
#include <set>
#include <cassert>
#include <map>
#include <vector>
#include <queue>
#include <algorithm>

namespace pass {
bool LoopParallel::isConstant(ir::Value* val) {
  if (val->isa<ir::ConstantValue>() or val->isa<ir::GlobalVariable>()) {
    return true;
  }

  // if(auto inst = val->dynCast<ir::Instruction>()){

  // }
  return false;
}
/**
 * void parallelFor(int32_t beg, int32_t end, void (*)(int32_t beg, int32_t end) func);
 *
 * void @parallelFor(i32 %beg, i32 %end, void (i32, i32)* %parallel_body_ptr);
 */
ir::Function* LoopParallel::loopupParallelFor(ir::Module* module) {
  if (auto func = module->findFunction("parallelFor")) {
    return func;
  }
  const auto voidType = ir::Type::void_type();
  const auto i32 = ir::Type::TypeInt32();
  // const auto funptrType = ir::PointerType::gen();

  // const auto parallelForType = ir::FunctionType::gen(
  //   ir::Type::void_type(), {i32, i32, ir::PointerType::gen(ir::Type::TypeInt8())});

  const auto parallelBodyPtrType =
    ir::PointerType::gen(ir::FunctionType::gen(voidType, {i32, i32}));

  const auto parallelForType = ir::FunctionType::gen(voidType, {i32, i32, parallelBodyPtrType});

  auto parallelFor = module->addFunction(parallelForType, "parallelFor");

  return parallelFor;
}

void LoopParallel::run(ir::Function* func, TopAnalysisInfoManager* tp) {
  runImpl(func, tp);
}

void LoopParallel::runImpl(ir::Function* func, TopAnalysisInfoManager* tp) {
  func->rename();
  func->print(std::cerr);

  CFGAnalysisHHW().run(func, tp);  // refresh CFG

  auto lpctx = tp->getLoopInfo(func);         // fisrt loop analysis
  auto indVarInfo = tp->getIndVarInfo(func);  // then indvar analysis

  // for all loops
  for (auto loop : lpctx->loops()) {
    loop->print(std::cerr);
    const auto indVar = indVarInfo->getIndvar(loop);
    const auto step = indVar->getStep()->i32();
    indVar->print(std::cerr);

    if (step != 1) continue;  // only support step = 1

    // extact loop body as a new loop_body func from func.loop
    /**
     * before:
     * entry -> body/header -> latch -> exit
     * after:
     * entry -> newLoop{phi, call loop_body, i.next, icmp, br} -> exit
     */
    LoopBodyFuncInfo loopBodyInfo;
    if (not extractLoopBody(func, *loop /* modified */, indVar, tp, loopBodyInfo /* ret */))
      continue;

    auto parallelFor = loopupParallelFor(func->module());

    const auto i32 = ir::Type::TypeInt32();
    const auto f32 = ir::Type::TypeFloat32();

    auto parallelBodyType = ir::FunctionType::gen(ir::Type::void_type(), {i32, i32});
    auto parallelBody = func->module()->addFunction(parallelBodyType, "parallelBody");

    // prepare payload for loop_body call
    std::vector<std::pair<ir::Value*, size_t>> payload;
    size_t totalSize = 0;
    size_t maxAlignment = 0;
    std::unordered_set<ir::Value*> insertedArgs;
    auto addArg = [&](Value* realArg) {
      if (realArg == indVar->phiinst()) return;
      if (realArg->isa<ir::ConstantValue>()) return;
      // if(realArg == loopBodyInfo.rec) return; ?
      if (not insertedArgs.insert(realArg).second) return;  // avoid duplicate args
      const auto align = 4;                                 // 4 or 8, what is the difference?
      const auto size = realArg->type()->size();
      totalSize = (totalSize + align - 1) / align * align;
      // TODO: handle giv
      payload.emplace_back(realArg, totalSize);
      totalSize += size;
    };

    for (auto use : loopBodyInfo.callInst->rargs()) {
      auto real_arg = use->value();
      addArg(real_arg);
    }

    const auto zero = ir::ConstantInteger::gen_i32(0);
    // arrayType?
    const auto playloadStorage =
      ir::GlobalVariable::gen(i32, {zero}, func->module(), "payloadStorage");

    ir::IRBuilder builder;
    // build parallelBody function
    const auto beg = parallelBody->new_arg(i32);
    const auto end = parallelBody->new_arg(i32);

    const auto entry = parallelBody->newEntry();
    const auto subLoop = parallelBody->newBlock();
    const auto exit = parallelBody->newExit();

    // build parallelBody.entry
    builder.set_pos(entry);
    const auto loopBodyCall = loopBodyInfo.callInst->clone();  // clone new call inst
    auto parallelBodyIndVar = builder.makeIdenticalInst<ir::PhiInst>(nullptr, i32);
    parallelBodyIndVar->addIncoming(beg, entry);
    // giv

    auto remapArgument = [&](ir::Use* use) {
      if (use->value() == indVar->phiinst()) {
        use->set_value(parallelBodyIndVar);
      }
      // else if(giv)
      else {
        if (isConstant(use->value())) {
          // expandConstant
          return;
        } else {
          // load from payloadStorage
        }
      }
    };
    for (auto opuse : loopBodyCall->operands()) {
      // const auto
      remapArgument(opuse);
    }
    // if(giv)
    builder.makeInst<ir::BranchInst>(subLoop);  // (entry) jump to subLoop

    // build parallelBody.subLoop
    subLoop->emplace_back_inst(parallelBodyIndVar);
    subLoop->emplace_back_inst(loopBodyCall);

    builder.set_pos(subLoop);

    // nexti = add i, 12
    const auto nexti =
      builder.makeBinary(ir::BinaryOp::ADD, parallelBodyIndVar, ir::ConstantInteger::gen_i32(1));
    parallelBodyIndVar->addIncoming(nexti, subLoop);

    // cond = icmp <, nexti, end
    const auto cond = builder.makeCmp(ir::CmpOp::LT, nexti, end);
    // br cond, exit, subLoop
    const auto br = builder.makeInst<ir::BranchInst>(cond, subLoop, exit);

    // build parallelBody.exit
    builder.set_pos(exit);

    // if(giv)
    builder.makeInst<ir::ReturnInst>();

    // set pos before call loop_body
    // builder.set_pos()
    // prepare giv
    // prepare payloadStorage
    // prepare parallel_body func ptr
    // call parallelFor(beg, end, parallel_body_ptr)
    // void (i32, i32)*
    const auto voidType = ir::Type::void_type();
    const auto parallelBodyPtrType =
      ir::PointerType::gen(ir::FunctionType::gen(voidType, {i32, i32}));

    auto parallelBodyFuncPtr =
      builder.makeInst<ir::FunctionPtrInst>(parallelBody, parallelBodyPtrType);

    auto callArgs = std::vector<ir::Value*>{loopBodyInfo.indVar->getBegin(),
                                            loopBodyInfo.indVar->getEnd(), parallelBodyFuncPtr};
    builder.makeInst<ir::CallInst>(parallelFor, callArgs);
  }
}

}  // namespace pass