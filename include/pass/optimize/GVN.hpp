#pragma once
#include <set>
#include <cassert>
#include <map>
#include <vector>
#include "ir/ir.hpp"
#include "pass/pass.hpp"

namespace pass {

class GVN : public FunctionPass {
  private:
    std::vector<ir::BasicBlock*> RPOblocks;
    std::set<ir::BasicBlock*> visited;
    std::set<ir::Instruction*> NeedRemove;
    std::unordered_map<ir::Value*, ir::Value*> _Hashtable;

  public:
    void run(ir::Function* func, TopAnalysisInfoManager* tp) override;
    void RPO(ir::Function* F);  // 逆后序遍历
    void dfs(ir::BasicBlock* bb);
    ir::Value* checkHashtable(ir::Value* v);
    ir::Value* getValueNumber(ir::Instruction* inst);
    ir::Value* getValueNumber(ir::BinaryInst* binary);
    ir::Value* getValueNumber(ir::UnaryInst* unary);
    ir::Value* getValueNumber(ir::GetElementPtrInst* getelementptr);
    ir::Value* getValueNumber(ir::LoadInst* load);
    // ir::Value *getValueNumber(ir::PhiInst *phi);
    // ir::Value *getValueNumber(ir::CallInst *cal);
    // ir::Value *getValueNumber(ir::AllocaInst *alloca);
    // ir::Value *getValueNumber(ir::ICmpInst *icmp);
    // ir::Value *getValueNumber(ir::FCmpInst *fcmp);
    void visitinst(ir::Instruction* inst);
    domTree* domctx;
};
}  // namespace pass