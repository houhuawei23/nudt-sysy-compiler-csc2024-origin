#include "pass/optimize/optimize.hpp"
#include "pass/optimize/Loop/LoopParallel.hpp"
#include "pass/analysis/ControlFlowGraph.hpp"
#include "pass/optimize/Loop/LoopBodyExtract.hpp"
#include "pass/optimize/Loop/ParallelBodyExtract.hpp"
#include "pass/optimize/Utils/BlockUtils.hpp"
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

  const auto parallelBodyPtrType = ir::FunctionType::gen(voidType, {i32, i32});

  const auto parallelForType = ir::FunctionType::gen(voidType, {i32, i32, parallelBodyPtrType});

  auto parallelFor = module->addFunction(parallelForType, "parallelFor");

  return parallelFor;
}

void LoopParallel::run(ir::Function* func, TopAnalysisInfoManager* tp) {
  runImpl(func, tp);
}

bool LoopParallel::runImpl(ir::Function* func, TopAnalysisInfoManager* tp) {
  func->rename();
  // func->print(std::cerr);

  CFGAnalysisHHW().run(func, tp);  // refresh CFG

  auto lpctx = tp->getLoopInfo(func);         // fisrt loop analysis
  auto indVarInfo = tp->getIndVarInfo(func);  // then indvar analysis
  bool modified = false;
  // for all loops
  lpctx->print(std::cerr);
  for (auto loop : lpctx->loops()) {
    // loop->print(std::cerr);
    const auto indVar = indVarInfo->getIndvar(loop);
    const auto step = indVar->getStep()->i32();
    // indVar->print(std::cerr);

    if (step != 1) continue;  // only support step = 1
    ParallelBodyInfo parallelBodyInfo;
    if (not extractParallelBody(func, loop /* modified */, indVar, tp, parallelBodyInfo /* ret */))
      continue;
    std::cerr << "parallel body extracted" << std::endl;
    // func->print(std::cerr);
    const auto parallelBody = parallelBodyInfo.parallelBody;
    auto parallelFor = loopupParallelFor(func->module());

    const auto i32 = ir::Type::TypeInt32();
    const auto f32 = ir::Type::TypeFloat32();
    IRBuilder builder;
    const auto callBlock = parallelBodyInfo.callBlock;
    auto& insts = parallelBodyInfo.callBlock->insts();
    std::vector<ir::Value*> args = {parallelBodyInfo.beg, parallelBodyInfo.end, parallelBody};
    for (auto iter = insts.begin(); iter != insts.end(); ++iter) {
      const auto inst = *iter;
      // inst->print(std::cerr);
      // std::cerr << std::endl;
      if (inst == parallelBodyInfo.callInst) {
        builder.set_pos(callBlock, iter);
        builder.makeInst<CallInst>(parallelFor, args);
        callBlock->move_inst(parallelBodyInfo.callInst);  // remove call parallel_body
        break;
      }
    }

    const auto fixFunction = [&](Function* function) {
      CFGAnalysisHHW().run(function, tp);
      blockSortDFS(*function, tp);
      function->rename();
      // function->print(std::cerr);
    };
    fixFunction(func);
    modified = true;
    break;
  }
  return modified;
}

}  // namespace pass