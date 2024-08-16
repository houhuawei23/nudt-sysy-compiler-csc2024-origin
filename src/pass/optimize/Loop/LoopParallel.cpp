// #define DEBUG

#include "pass/optimize/optimize.hpp"
#include "pass/optimize/Loop/LoopParallel.hpp"
#include "pass/analysis/ControlFlowGraph.hpp"
#include "pass/optimize/Loop/LoopBodyExtract.hpp"
#include "pass/optimize/Loop/ParallelBodyExtract.hpp"
#include "pass/optimize/Utils/BlockUtils.hpp"
#include "pass/analysis/MarkParallel.hpp"
#include <set>
#include <cassert>
#include <map>
#include <vector>
#include <queue>
#include <algorithm>

using namespace ir;
namespace pass {
bool LoopParallel::isConstant(Value* val) {
  if (val->isa<ConstantValue>() or val->isa<GlobalVariable>()) {
    return true;
  }

  // if(auto inst = val->dynCast<Instruction>()){

  // }
  return false;
}
/**
 * void parallelFor(int32_t beg, int32_t end, void (*)(int32_t beg, int32_t end) func);
 *
 * void @parallelFor(i32 %beg, i32 %end, void (i32, i32)* %parallel_body_ptr);
 */
static Function* loopupParallelFor(Module* module) {
  if (auto func = module->findFunction("parallelFor")) {
    return func;
  }
  const auto voidType = Type::void_type();
  const auto i32 = Type::TypeInt32();

  const auto parallelBodyPtrType = FunctionType::gen(voidType, {i32, i32});

  const auto parallelForType = FunctionType::gen(voidType, {i32, i32, parallelBodyPtrType});

  auto parallelFor = module->addFunction(parallelForType, "parallelFor");

  return parallelFor;
}

void LoopParallel::run(Function* func, TopAnalysisInfoManager* tp) {
  runImpl(func, tp);
}

bool parallelLoop(Function* func, TopAnalysisInfoManager* tp, Loop* loop, IndVar* indVar) {
  ParallelBodyInfo parallelBodyInfo;
  if (not extractParallelBody(func, loop /* modified */, indVar, tp, parallelBodyInfo /* ret */))
    return false;
  // std::cerr << "parallel body extracted" << std::endl;
  // func->print(std::cerr);
  const auto parallelBody = parallelBodyInfo.parallelBody;
  auto parallelFor = loopupParallelFor(func->module());

  IRBuilder builder;
  const auto callBlock = parallelBodyInfo.callBlock;
  auto& insts = parallelBodyInfo.callBlock->insts();
  std::vector<Value*> args = {parallelBodyInfo.beg, parallelBodyInfo.end, parallelBody};

  const auto iter = std::find(insts.begin(), insts.end(), parallelBodyInfo.callInst);
  assert(iter != insts.end());  // must find

  builder.set_pos(callBlock, iter);
  builder.makeInst<CallInst>(parallelFor, args);
  callBlock->move_inst(parallelBodyInfo.callInst);  // remove call parallel_body

  const auto fixFunction = [&](Function* function) {
    CFGAnalysisHHW().run(function, tp);
    blockSortDFS(*function, tp);
    // function->rename();
    // function->print(std::cerr);
  };
  fixFunction(func);
  return true;
}

bool LoopParallel::runImpl(Function* func, TopAnalysisInfoManager* tp) {
  func->rename();
  // func->print(std::cerr);

  CFGAnalysisHHW().run(func, tp);  // refresh CFG
  markParallel().run(func, tp);

  auto lpctx = tp->getLoopInfoWithoutRefresh(func);         // fisrt loop analysis
  auto indVarInfo = tp->getIndVarInfoWithoutRefresh(func);  // then indvar analysis
  auto parctx = tp->getParallelInfo(func);

  auto loops = lpctx->sortedLoops();

  bool modified = false;

  std::unordered_set<Loop*> extractedLoops;
  const auto isBlocked = [&](Loop* lp) {
    for (auto extracted : extractedLoops) {
      if (extracted->blocks().count(lp->header())) {
#ifdef DEBUG
        lp->header()->dumpAsOpernd(std::cerr);
        std::cerr << "is sub of ";
        extracted->header()->dumpAsOpernd(std::cerr);
        std::cerr << std::endl;
#endif
        return true;
      }
    }
    return false;
  };
  const auto checkLoop = [&](Loop* loop, IndVar* indVar) {
    if (lpctx->looplevel(loop->header()) > 2) {  // only consider loops with level <= 2
      // std::cerr << "loop level: " << lpctx->looplevel(loop->header());
      // std::cerr << " is too deep, skip" << std::endl;
      return false;
    }
    if (isBlocked(loop)) return false;
    if (not parctx->getIsParallel(loop->header())) {
      // std::cerr << "cant parallel" << std::endl;
      return false;
    }
    if (indVar == nullptr) {
      // std::cerr << "no indvar for loop: " << loop->header()->name() << std::endl;
      return false;
    }
    const auto step = indVar->getStep()->i32();
    if (step != 1) return false;  // only support step = 1

    if (indVar->beginValue()->isa<ConstantValue>() and indVar->endValue()->isa<ConstantValue>()) {
      const auto begin = indVar->beginValue()->dynCast<ConstantValue>()->i32();
      const auto end = indVar->endValue()->dynCast<ConstantValue>()->i32();
      if (end - begin < 100) {
        std::cerr << "loop too small: " << end - begin << std::endl;
        return false;
      }
    }
    return true;
  };

  // lpctx->print(std::cerr);
  for (auto loop : loops) {  // for all loops
    const auto indVar = indVarInfo->getIndvar(loop);
    if(not checkLoop(loop, indVar)) continue;
#ifdef DEBUG
    std::cerr << "loop level: " << lpctx->looplevel(loop->header());
    loop->print(std::cerr);
    indVar->print(std::cerr);
#endif
    auto success = parallelLoop(func, tp, loop, indVar);
    modified |= success;
    if (success) extractedLoops.insert(loop);
  }

  return modified;
}

}  // namespace pass