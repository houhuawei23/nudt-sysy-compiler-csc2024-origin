#pragma once
#include "pass/pass.hpp"

namespace pass {

class domInfoPass : public FunctionPass {
  public:
    void run(ir::Function* func, TopAnalysisInfoManager* tp) override;
    std::string name() const override { return "DomInfoPass"; }
};

class preProcDom : public FunctionPass {
  private:
    domTree* domctx;

  public:
    void run(ir::Function* func, TopAnalysisInfoManager* tp) override;
    std::string name() const override { return "PreProcDom"; }
};

class idomGen : public FunctionPass {
  private:
    domTree* domctx;

  private:
    void dfsBlocks(ir::BasicBlock* bb);
    ir::BasicBlock* eval(ir::BasicBlock* bb);
    void link(ir::BasicBlock* v, ir::BasicBlock* w);
    void compress(ir::BasicBlock* bb);

  public:
    void run(ir::Function* func, TopAnalysisInfoManager* tp) override;
    std::string name() const override { return "IdomGen"; }
};

class domFrontierGen : public FunctionPass {
  private:
    domTree* domctx;

  private:
    void getDomTree(ir::Function* func);
    void getDomFrontier(ir::Function* func);
    void getDomInfo(ir::BasicBlock* bb, int level);

  public:
    void run(ir::Function* func, TopAnalysisInfoManager* tp) override;
    std::string name() const override { return "DomFrontierGen"; }
};

class domInfoCheck : public FunctionPass {
  private:
    domTree* domctx;

  public:
    void run(ir::Function* func, TopAnalysisInfoManager* tp) override;
    std::string name() const override { return "DomInfoCheck"; }
};

}  // namespace pass
