#pragma once
#include "ir/ir.hpp"
#include "pass/pass.hpp"
#include <vector>
#include <set>

namespace pass {
class callGraphBuild : public ModulePass {
public:
  std::string name() const override { return "callGraphBuild"; }
  void run(ir::Module* ctx, TopAnalysisInfoManager* tp) override;

private:
  std::vector<ir::Function*> funcStack;
  std::set<ir::Function*> funcSet;
  std::map<ir::Function*, bool> vis;
  void dfsFuncCallGraph(ir::Function* func);

private:
  callGraph* cgctx;
};

class callGraphCheck : public ModulePass {
public:
  std::string name() const override { return "callGraphCheck"; }
  void run(ir::Module* ctx, TopAnalysisInfoManager* tp) override;

private:
  callGraph* cgctx;
};
}  // namespace pass