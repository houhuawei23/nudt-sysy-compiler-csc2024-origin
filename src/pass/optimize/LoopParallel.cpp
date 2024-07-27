#include "pass/optimize/optimize.hpp"
#include "pass/optimize/LoopParallel.hpp"
#include <set>
#include <cassert>
#include <map>
#include <vector>
#include <queue>
#include <algorithm>

namespace pass {

void LoopParallel::run(ir::Function* func,TopAnalysisInfoManager* tp) {
  runImpl(func);
}
void LoopParallel::runImpl(ir::Function* func) {}

}  // namespace pass