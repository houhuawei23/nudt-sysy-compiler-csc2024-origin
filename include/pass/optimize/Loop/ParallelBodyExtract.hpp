#include <set>
#include <cassert>
#include <map>
#include <vector>
#include <queue>
#include "ir/ir.hpp"
#include "pass/pass.hpp"

using namespace ir;
namespace pass {

class ParallelBodyExtract : public FunctionPass {
public:
  void run(ir::Function* func, TopAnalysisInfoManager* tp) override;
  std::string name() const override { return "ParallelBodyExtract"; }

private:
  void runImpl(ir::Function* func, TopAnalysisInfoManager* tp);
};



}  // namespace pass