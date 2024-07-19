#include <set>
#include <cassert>
#include <map>
#include <vector>
#include "ir/ir.hpp"
#include "pass/pass.hpp"

namespace pass {
class LCSSA : public FunctionPass {
  private:
  domTree* DT;
  public:
  bool isLCSSAform(ir::Loop* L);
  ir::Value* stack_pop(std::stack<ir::Value*>& stack);
  void rename(std::stack<ir::Value*>& stack, ir::BasicBlock* bb, std::map<ir::Instruction*, int> useInstrMap, std::set<ir::Instruction*> defInstrs);
  void addDef(std::set<ir::Instruction*>& definsts, ir::Instruction* inst);
  void makeExitPhi(ir::Instruction* inst, ir::BasicBlock* exit, ir::Loop* L);
  bool isUseout(ir::Instruction* inst, ir::Loop* L);
  void runonloop(ir::Loop* L,
                 loopInfo* LI,
                 topAnalysisInfoManager* tp);
  void run(ir::Function* func, topAnalysisInfoManager* tp) override;
};
}  // namespace pass