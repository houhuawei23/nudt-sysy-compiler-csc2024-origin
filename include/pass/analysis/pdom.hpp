#pragma once
#include "pass/pass.hpp"

namespace pass {

class postDomInfoPass : public FunctionPass {
public:
  void run(ir::Function* func, TopAnalysisInfoManager* tp) override;
  std::string name() const override { return "PostDomInfoPass"; }

private:
  PDomTree* pdctx;
};

class preProcPostDom : public FunctionPass {
public:
  void run(ir::Function* func, TopAnalysisInfoManager* tp) override;
  std::string name() const override { return "preProcPostDom"; }

private:
  PDomTree* pdctx;
};

class ipostDomGen : public FunctionPass {
private:
  void dfsBlocks(ir::BasicBlock* bb);
  ir::BasicBlock* eval(ir::BasicBlock* bb);
  void link(ir::BasicBlock* v, ir::BasicBlock* w);
  void compress(ir::BasicBlock* bb);

public:
  void run(ir::Function* func, TopAnalysisInfoManager* tp) override;
  std::string name() const override { return "ipostDomGen"; }

private:
  PDomTree* pdctx;
};

class postDomFrontierGen : public FunctionPass {
private:
  void getDomTree(ir::Function* func);
  void getDomFrontier(ir::Function* func);
  void getDomInfo(ir::BasicBlock* bb, int level);

public:
  void run(ir::Function* func, TopAnalysisInfoManager* tp) override;
  std::string name() const override { return "postDomFrontierGen"; }

private:
  PDomTree* pdctx;
};

class postDomInfoCheck : public FunctionPass {
public:
  void run(ir::Function* func, TopAnalysisInfoManager* tp) override;
  std::string name() const override { return "postDomInfoCheck"; }

private:
  PDomTree* pdctx;
};

}  // namespace pass
