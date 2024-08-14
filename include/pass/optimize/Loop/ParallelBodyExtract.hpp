#include <set>
#include <cassert>
#include <map>
#include <vector>
#include <queue>
#include "ir/ir.hpp"
#include "pass/pass.hpp"

using namespace ir;
namespace pass {

struct ParallelBodyInfo final {
  Function* parallelBody;
  CallInst* callInst;
  BasicBlock* callBlock;
  Value* beg;
  Value* end;
};

class ParallelBodyExtract : public FunctionPass {
public:
  void run(ir::Function* func, TopAnalysisInfoManager* tp) override;
  std::string name() const override { return "ParallelBodyExtract"; }

private:
  bool runImpl(ir::Function* func, TopAnalysisInfoManager* tp);
};

bool extractParallelBody(Function* func,
                         Loop* loop,
                         IndVar* indVar,
                         TopAnalysisInfoManager* tp,
                         ParallelBodyInfo& parallelBodyInfo);

}  // namespace pass