#pragma once
#include <set>
#include <cassert>
#include <map>
#include <vector>
#include "ir/ir.hpp"
#include "pass/pass.hpp"
namespace pass {
class loopUnroll : public FunctionPass {
  private:
    loopInfo* lpctx;
    indVarInfo* ivctx;
    std::unordered_map<ir::Value*, ir::Value*> copymap;
  public:
    std::string name() const override { return "loopunroll"; }
    bool definuseout(ir::Instruction* inst, ir::Loop* L);
    bool defoutusein(ir::Use* op, ir::Loop* L);
    void copyloop(std::vector<ir::BasicBlock*> bbs, ir::BasicBlock* begin, ir::Loop* L, ir::Function* func);
    int calunrolltime(ir::Loop* loop, int times);
    void doconstunroll(ir::Loop* loop, ir::indVar* iv, int times);
    void dynamicunroll(ir::Loop* loop, ir::indVar* iv);
    void constunroll(ir::Loop* loop, ir::indVar* iv);
    bool isconstant(ir::indVar* iv);
    void run(ir::Function* func, TopAnalysisInfoManager* tp) override;
};
}  // namespace pass