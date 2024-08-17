#include "pass/optimize/Loop/LoopUtils.hpp"
namespace pass {
bool checkLoopParallel(Loop* loop,
                       loopInfo* lpctx,
                       indVarInfo* indVarctx,
                       parallelInfo* parallelctx,
                       std::unordered_set<Loop*>& extractedLoops) {
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
  const auto indVar = indVarctx->getIndvar(loop);
  if (lpctx->looplevel(loop->header()) > 2) {  // only consider loops with level <= 2
    // std::cerr << "loop level: " << lpctx->looplevel(loop->header());
    // std::cerr << " is too deep, skip" << std::endl;
    return false;
  }
  if (isBlocked(loop)) return false;
  if (not parallelctx->getIsParallel(loop->header())) {
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
}
}  // namespace pass
