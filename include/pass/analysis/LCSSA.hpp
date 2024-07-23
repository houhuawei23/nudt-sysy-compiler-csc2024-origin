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
  void promoteLCSSA(ir::Instruction* usee, ir::Loop* L);
  void rename(ir::Instruction* Inst, ir::Loop* L, ir::Function* F, std::vector<ir::BasicBlock*> DefsBlock, std::vector<ir::Instruction*> Defs, std::vector<ir::PhiInst*> newphis);
  std::vector<ir::PhiInst*> insertphi(ir::Instruction* Inst, ir::Loop* L, std::vector<ir::BasicBlock*> &DefsBlock,std::vector<ir::Instruction*> &Defs);
  bool isUseout(ir::Instruction* inst, ir::Loop* L);
  void runonloop(ir::Loop* L,
                 loopInfo* LI,
                 topAnalysisInfoManager* tp);
  void run(ir::Function* func, topAnalysisInfoManager* tp) override;
};
}  // namespace pass