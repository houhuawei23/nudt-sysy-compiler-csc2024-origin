#define DEBUG

#include "pass/optimize/Loop/LoopInterChange.hpp"
#include "pass/optimize/Loop/LoopBodyExtract.hpp"
#include "pass/optimize/Loop/LoopUtils.hpp"

#include "pass/optimize/Utils/BlockUtils.hpp"
#include "pass/analysis/MarkParallel.hpp"
#include "pass/analysis/ControlFlowGraph.hpp"

using namespace ir;
using namespace pass;

bool detectPatternAndSwap(Function* func,
                          Loop* innerLoop,
                          loopInfo* lpctx,
                          indVarInfo* indVarctx,
                          TopAnalysisInfoManager* tp);

void LoopInterChange::run(ir::Function* func, TopAnalysisInfoManager* tp) {
  runImpl(func, tp);
}

bool LoopInterChange::runImpl(ir::Function* func, TopAnalysisInfoManager* tp) {
  func->rename();
  // func->print(std::cerr);

  CFGAnalysisHHW().run(func, tp);  // refresh CFG
  markParallel().run(func, tp);

  auto lpctx = tp->getLoopInfoWithoutRefresh(func);        // fisrt loop analysis
  auto indVarctx = tp->getIndVarInfoWithoutRefresh(func);  // then indvar analysis
  auto parallelctx = tp->getParallelInfo(func);

  auto loops = lpctx->sortedLoops();

  bool modified = false;

  std::unordered_set<Loop*> extractedLoops;

  // lpctx->print(std::cerr);
  for (auto loop : loops) {  // for all loops
    const auto indVar = indVarctx->getIndvar(loop);
    if (not checkLoopParallel(loop, lpctx, indVarctx, parallelctx, extractedLoops)) continue;
#ifdef DEBUG
    std::cerr << "loop level: " << lpctx->looplevel(loop->header());
    loop->print(std::cerr);
    indVar->print(std::cerr);
#endif
    // auto success = detectPatternAndSwap(loop);
    auto success = false;
    modified |= success;
    if (success) extractedLoops.insert(loop);
  }

  return modified;
}

/*
    current loop is inner loop.

---> outer exit
|
|   outer preheader
|       |
--- outer header <-----------
        |                   |
    inner preheader         |
        |                   |
--> inner header --> outer latch
|       |
|   call loop_body
|       |
--- inner latch

==>

--> outer exit
|
|   outer preheader
|       |
--- inner header <-----------
        |                   |
--> outer header --> inner latch
|       |
|   inner preheader
|       |
|   call loop_body
|       |
--- outer latch



*/
bool detectPatternAndSwap(Function* func,
                          Loop* innerLoop,
                          loopInfo* lpctx,
                          indVarInfo* indVarctx,
                          TopAnalysisInfoManager* tp) {
  CFGAnalysisHHW().run(func, tp);  // refresh CFG
  const auto outerLoop = innerLoop->parentloop();
  if (outerLoop == nullptr) return false;

  const auto outerLever = lpctx->looplevel(outerLoop->header());
  const auto innerLever = lpctx->looplevel(innerLoop->header());
  if (not(outerLever == 1 and innerLever == 2)) return false;

  // 1 extract inner loop body
  const auto innerIndVar = indVarctx->getIndvar(innerLoop);
  const auto outerIndVar = indVarctx->getIndvar(outerLoop);
  if (innerIndVar == nullptr or outerIndVar == nullptr) return false;

  LoopBodyInfo innerBodyInfo;
  if (not extractLoopBody(func, *innerLoop, innerIndVar, tp, innerBodyInfo)) return false;

#ifdef DEBUG
  innerBodyInfo.print(std::cerr);
  func->print(std::cerr);
  innerBodyInfo.callInst->callee()->print(std::cerr);
#endif

  // pattern match
  // 1 innerLoop.preheader in outerLoop.header.next_blocks()
  const auto innerPreheader = innerLoop->getLoopPreheader();
  const auto outerHeader = outerLoop->header();
  const auto iter =
    std::find(outerHeader->next_blocks().begin(), outerHeader->next_blocks().end(), innerPreheader);
  if (iter == outerHeader->next_blocks().end()) return false;

  // 2 innerLoop.exit == outerLoop.latch
  if (innerLoop->exits().size() != 1) return false;
  if (outerLoop->exits().size() != 1) return false;
  const auto innerExit = *(innerLoop->exits().begin());
  const auto outerExit = *(outerLoop->exits().begin());
  if (innerLoop->latchs().size() != 1 and outerLoop->latchs().size() != 1) return false;
  const auto innerLatch = innerLoop->getUniqueLatch();
  const auto outerLatch = outerLoop->getUniqueLatch();

  if (innerExit != outerLatch) return false;

  // manual clone outerLoop.indVar.iterInst to outerLoop.latch
  IRBuilder builder;
  // rebuild branch relationship, fix phi insts
  // 1 outer preheader -> inner header
  const auto outerPreheader = outerLoop->getLoopPreheader();
  const auto innerHeader = innerLoop->header();
  outerPreheader->insts().pop_back();  // remove br outer header
  builder.setInsetPosEnd(outerPreheader);
  builder.makeInst<BranchInst>(innerHeader);

  // 2 inner header: br cond, outer header, outer exit
  innerHeader->insts().pop_back();  // remove br inner call loop_body
  builder.setInsetPosEnd(innerHeader);
  const auto innerCmpInst = innerIndVar->cmpInst()->dynCast<ICmpInst>();
  if (innerCmpInst == nullptr) {
    assert(false && "inner cmp inst is null");
    return false;
  }
  builder.makeInst<BranchInst>(innerCmpInst, outerHeader, outerExit);

  // 3 outer header: br outerCmpInst, inner preheader, inner latch
  outerHeader->insts().pop_back();  // remove br inner preheader
  const auto outerCmpInst = outerIndVar->cmpInst()->dynCast<ICmpInst>();
  if (outerCmpInst == nullptr) {
    assert(false && "outer cmp inst is null");
    return false;
  }
  builder.setInsetPosEnd(outerHeader);
  builder.makeInst<BranchInst>(outerCmpInst, innerPreheader, innerLatch);

  // 4 inner preheader: br call loop_body
  const auto innerBody = innerBodyInfo.callInst->block();  // innerBodyInfo->body?

  innerPreheader->insts().pop_back();  // remove br inner header
  builder.setInsetPosEnd(innerPreheader);
  builder.makeInst<BranchInst>(innerBody);

  // 5 call loop_body: br outer latch
  innerBody->insts().pop_back();  // remove br inner latch
  builder.setInsetPosEnd(innerBody);
  builder.makeInst<BranchInst>(outerLatch);

  // 6 outer latch: br outer header, None
  // 7 inner latch: br inner header, None

  // refresh CFG
  CFGAnalysisHHW().run(func, tp);  // refresh CFG
  return true;
}