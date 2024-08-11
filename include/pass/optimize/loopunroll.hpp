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
    static std::unordered_map<ir::Value*, ir::Value*> copymap;
    std::vector<ir::Instruction*> headuseouts;
    ir::BasicBlock* nowlatchnext;
  public:
    std::string name() const override { return "loopunroll"; }
    bool definuseout(ir::Instruction* inst, ir::Loop* L);
    void insertremainderloop(ir::Loop* loop, ir::Function* func);
    void copyloop(std::vector<ir::BasicBlock*> bbs, ir::BasicBlock* begin, ir::Loop* L, ir::Function* func);
    void copyloopremainder(std::vector<ir::BasicBlock*> bbs, ir::BasicBlock* begin, ir::Loop* L, ir::Function* func);
    int calunrolltime(ir::Loop* loop, int times);
    void doconstunroll(ir::Loop* loop, ir::IndVar* iv, int times);
    void dodynamicunroll(ir::Loop* loop, ir::IndVar* iv);
    void dynamicunroll(ir::Loop* loop, ir::IndVar* iv);
    void constunroll(ir::Loop* loop, ir::IndVar* iv);
    void dofullunroll(ir::Loop* loop, ir::IndVar* iv, int times);
    bool isconstant(ir::IndVar* iv);
    void getdefinuseout(ir::Loop* L);
    void repalceuseout(ir::Instruction* inst, ir::Instruction* copyinst, ir::Loop* L);
    static ir::Value* getValue(ir::Value* val) {
        if (auto c = val->dynCast<ir::ConstantValue>()) {
            return c;
        }
        if (copymap.count(val)) {
            return copymap[val];
        }
        return val;
    }
    void run(ir::Function* func, TopAnalysisInfoManager* tp) override;
};
}  // namespace pass