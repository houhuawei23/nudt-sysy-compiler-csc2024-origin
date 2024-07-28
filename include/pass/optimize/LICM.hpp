#pragma once
#include <set>
#include <cassert>
#include <map>
#include <vector>
#include "ir/ir.hpp"
#include "pass/pass.hpp"
namespace pass {

class LICM : public ModulePass {
private:
  std::vector<ir::AllocaInst*> Allallocas;
  std::unordered_map<ir::AllocaInst*, std::set<ir::Instruction*>> allocaDefs;
  std::unordered_map<ir::AllocaInst*, std::set<ir::Instruction*>> allocaUses;
  std::unordered_map<ir::Value*, std::set<ir::Instruction*>> gDefs;
  std::unordered_map<ir::Value*, std::set<ir::Instruction*>> gUses;
  std::unordered_map<ir::Function*, std::set<ir::Value*>> funcDefGlobals;
  std::unordered_map<ir::Function*, std::set<ir::Value*>> funcUseGlobals;
  std::unordered_map<ir::Value*, std::unordered_map<ir::Value*, ir::Loop*>>
    useLoops;
  std::unordered_map<ir::Value*, std::set<ir::Loop*>> defLoops;
  std::unordered_map<ir::Function*, loopInfo*> flmap;
  std::vector<ir::Function*> funcList;
  callGraph* cgctx;
  loopInfo* loopctx;

public:
  std::string name() const override { return "LICM"; }
  void clear();
  void getallallocas(ir::Function* func);
  bool check(ir::Loop* innerL, ir::Loop* defL);
  bool storemove(ir::AllocaInst* alloca);
  void storeLift(ir::Module* module);
  void storeLiftfunc(ir::Function* func);
  bool loadmove(ir::Value* val);
  void loadLift(ir::Module* module);
  void dfs(ir::AllocaInst* alloca, ir::Instruction* inst);
  void globaldfs(ir::Value* val, ir::Instruction* inst);
  std::vector<ir::Value*> keyset(
    std::unordered_map<ir::Value*, std::set<ir::Instruction*>> map);
  void run(ir::Module* module, TopAnalysisInfoManager* tp) override;
};
}  // namespace pass