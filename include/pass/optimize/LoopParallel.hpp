#include <set>
#include <cassert>
#include <map>
#include <vector>
#include <queue>
#include "ir/ir.hpp"
#include "pass/pass.hpp"

namespace pass {
class LoopParallel : public FunctionPass {
public:
    void run(ir::Function* func, topAnalysisInfoManager* tp) override;
    std::string name() { return "LoopParallel"; }

private:
    void runImpl(ir::Function* func);
};
}  // namespace pass