#include <set>
#include <cassert>
#include <map>
#include <vector>
#include <queue>
#include "ir/ir.hpp"
#include "pass/pass.hpp"

using namespace ir;
namespace pass {

struct LoopBodyFuncInfo {
  BasicBlock* loop;
  CallInst* callInst;
  ir::indVar* indVar;
};

class LoopParallel : public FunctionPass {
public:
  void run(ir::Function* func, TopAnalysisInfoManager* tp) override;
  std::string name() { return "LoopParallel"; }

private:
  static ir::Function* loopupParallelFor(ir::Module* module);
  static bool isConstant(ir::Value* val);
  void runImpl(ir::Function* func, TopAnalysisInfoManager* tp);
};

bool extractLoopBody(ir::Function* func,
                     ir::Loop& loop,
                     ir::indVar* indVar,
                     TopAnalysisInfoManager* tp,
                     LoopBodyFuncInfo& loopBodyInfo);

}  // namespace pass