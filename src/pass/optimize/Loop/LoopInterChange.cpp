#include "pass/optimize/Loop/LoopInterChange.hpp"
#include "pass/optimize/Utils/BlockUtils.hpp"

using namespace ir;
using namespace pass;

void LoopInterChange::run(ir::Function* func, TopAnalysisInfoManager* tp) {
  runImpl(func, tp);
}

bool LoopInterChange::runImpl(ir::Function* func, TopAnalysisInfoManager* tp) {


  return true;
}
