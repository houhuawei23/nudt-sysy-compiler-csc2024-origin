#include "pass/optimize/Loop/ParallelBodyExtract.hpp"
#include "pass/optimize/Loop/LoopBodyExtract.hpp"
#include "pass/analysis/ControlFlowGraph.hpp"
#include "pass/optimize/Utils/BlockUtils.hpp"

// #include "pass/analysis/dependenceAnalysis/dpaUtils.hpp"
#include "pass/analysis/dependenceAnalysis/DependenceAnalysis.hpp"
using namespace ir;

namespace pass {

static auto getUniqueID() {
  static size_t id = 0;
  const auto base = "sysyc_parallel_body";
  return base + std::to_string(id++);
}
/*
after extract loop body:
  preheader -> header -> call_block -> latch -> exit
               header <--------------- latch
  call_block: call loop_body(i, otherargs...)

after extract parallel body:
  preheader -> call_block -> exit
  call parallel_body(beg, end)

  parallel_body(beg, end)
    for (i = beg; i < end; i++) {
      loop_body(i, otherargs...)
    }

*/
bool extractParallelBody(Function* func,
                         Loop* loop,
                         IndVar* indVar,
                         TopAnalysisInfoManager* tp,
                         ParallelBodyInfo& parallelBodyInfo) {
  const auto step = indVar->getStep()->i32();
  // indVar->print(std::cerr);

  if (step != 1) return false;                  // only support step = 1
  if (loop->exits().size() != 1) return false;  // only support single exit loop
  const auto loopExitBlock = *(loop->exits().begin());
  // extact loop body as a new loop_body func from func loop
  LoopBodyInfo loopBodyInfo;
  if (not extractLoopBody(func, *loop /* modified */, indVar, tp, loopBodyInfo /* ret */))
    return false;

  // loopBodyInfo.print(std::cerr);
  const auto i32 = Type::TypeInt32();
  auto funcType = FunctionType::gen(Type::void_type(), {i32, i32});
  auto parallelBody = func->module()->addFunction(funcType, getUniqueID());

  auto argBeg = parallelBody->new_arg(i32, "beg");
  auto argEnd = parallelBody->new_arg(i32, "end");

  auto newEntry = parallelBody->newEntry("new_entry");
  auto newExit = parallelBody->newExit("new_exit");
  std::unordered_set<BasicBlock*> bodyBlocks = {loopBodyInfo.header, loopBodyInfo.body,
                                                loopBodyInfo.latch};
  // add loop blocks to parallel_body
  for (auto block : bodyBlocks) {
    block->set_parent(parallelBody);
    parallelBody->blocks().push_back(block);
  }
  // remove loop from func
  func->blocks().remove_if([&](BasicBlock* block) { return bodyBlocks.count(block); });
  // build parallel_body
  IRBuilder builder;
  builder.set_pos(newEntry, newEntry->insts().end());
  builder.makeInst<BranchInst>(loopBodyInfo.header);
  builder.set_pos(newExit, newExit->insts().end());
  builder.makeInst<ReturnInst>();

  // add new call block to func
  auto callBlock = func->newBlock();
  callBlock->setComment("call_parallel_body");
  loopBodyInfo.preHeader->insts().pop_back();  // remove old br from preHeader
  builder.set_pos(loopBodyInfo.preHeader, loopBodyInfo.preHeader->insts().end());
  builder.makeInst<BranchInst>(callBlock);
  builder.set_pos(callBlock, callBlock->insts().end());
  // call parallel_body(beg, end)
  auto callArgs = std::vector<Value*>{indVar->getBegin(), indVar->getEnd()};
  auto callInst = builder.makeInst<CallInst>(parallelBody, callArgs);
  assert(loop->exits().size() == 1);
  builder.makeInst<BranchInst>(loopExitBlock);

  // fix value in paraplel_body
  const auto fixPhi = [&](PhiInst* phi) {
    // std::cerr << "fix phi inst: ";
    // phi->dumpAsOpernd(std::cerr);
    // std::cerr << std::endl;
    if (phi == indVar->phiinst()) {
      phi->delBlock(loopBodyInfo.preHeader);
      phi->addIncoming(argBeg, newEntry);
      return;
    }
    // std::cerr << "phi inst not indvar phi inst" << std::endl;
    // phi->print(std::cerr);
    // std::cerr << std::endl;
  };

  // fix cmp inst
  const auto fixCmp = [&](ICmpInst* cmpInst) {
    if (cmpInst == indVar->cmpInst()) {
      for (auto opuse : cmpInst->operands()) {
        auto op = opuse->value();
        if (op == indVar->getEnd()) {
          cmpInst->setOperand(opuse->index(), argEnd);
          break;
        }
      }
    }
    // std::cerr << "cmp inst not indvar cmp inst" << std::endl;
    // cmpInst->print(std::cerr);
    // std::cerr << std::endl;
  };
  std::unordered_map<BasicBlock*, BasicBlock*> blockMap;
  blockMap.emplace(loopExitBlock, newExit);
  const auto fixBranch = [&](BranchInst* branch) {
    for (auto opuse : branch->operands()) {
      auto op = opuse->value();
      if (auto block = op->dynCast<BasicBlock>()) {
        if (auto pair = blockMap.find(block); pair != blockMap.end()) {
          branch->setOperand(opuse->index(), pair->second);
        }
      }
    }
  };
  std::vector<std::pair<Value*, size_t>> payload;
  // non constant, gloabal value used in loop_body, must pass by global payload
  const auto fixCall = [&](CallInst* call) {
    if (call == loopBodyInfo.callInst) {  // call loop_body(i, otherargs...)
      for (auto opuse : call->operands()) {
        auto op = opuse->value();
        // TODO:
      }
    }
  };
  for (auto block : parallelBody->blocks()) {
    // std::cerr << "block: " << block->name() << std::endl;
    for (auto inst : block->insts()) {
      // inst->print(std::cerr);
      // std::cerr << std::endl;
      if (auto phi = inst->dynCast<PhiInst>()) {
        fixPhi(phi);
      } else if (auto cmpInst = inst->dynCast<ICmpInst>()) {
        fixCmp(cmpInst);
      } else if (auto branch = inst->dynCast<BranchInst>()) {
        fixBranch(branch);
      } else if (auto call = inst->dynCast<CallInst>()) {
        fixCall(call);
      }
    }
  }
  const auto fixFunction = [&](Function* function) {
    CFGAnalysisHHW().run(function, tp);
    blockSortDFS(*function, tp);
    function->rename();
    // function->print(std::cerr);
  };
  // fic function
  fixFunction(func);
  fixFunction(parallelBody);
  parallelBodyInfo.parallelBody = parallelBody;
  parallelBodyInfo.callInst = callInst;
  parallelBodyInfo.callBlock = callBlock;
  parallelBodyInfo.beg = indVar->getBegin();
  parallelBodyInfo.end = indVar->getEnd();
  return true;
}

void ParallelBodyExtract::run(ir::Function* func, TopAnalysisInfoManager* tp) {
  runImpl(func, tp);
}

#define NDEBUG
bool ParallelBodyExtract::runImpl(ir::Function* func, TopAnalysisInfoManager* tp) {
  auto sideEffectInfo = tp->getSideEffectInfo();

  func->rename();
  // func->print(std::cerr);

  CFGAnalysisHHW().run(func, tp);  // refresh CFG

  bool modified = false;
  auto dpctx = tp->getDepInfo(func);
  auto lpctx = tp->getLoopInfoWithoutRefresh(func);         // fisrt loop analysis
  auto indVarInfo = tp->getIndVarInfoWithoutRefresh(func);  // then indvar analysis

  auto loops = lpctx->sortedLoops();
  std::unordered_set<Loop*> extractedLoops;
  const auto isBlocked = [&](Loop* lp) {
    for (auto extracted : extractedLoops) {
      if (extracted->blocks().count(lp->header())) {
#ifndef NDEBUG
        lp->header()->dumpAsOpernd(std::cerr);
        std::cerr << "is sub of ";
        extracted->header()->dumpAsOpernd(std::cerr);
        std::cerr << std::endl;
#endif
        return true;
      }
    }
    return false;
  };
  for (auto loop : loops) {  // for all loops
    // loop->print()
    if (isBlocked(loop)) continue;
    if (not dpctx->getLoopDependenceInfo(loop)->getIsParallel())continue;
    const auto indVar = indVarInfo->getIndvar(loop);
    const auto step = indVar->getStep()->i32();
    if (step != 1) continue;  // only support step = 1

    ParallelBodyInfo info;
    if (not extractParallelBody(func, loop, indVar, tp, info)) {
      // std::cerr << "failed to extract parallel body for loop" << std::endl;
      continue;
    }
    modified = true;
    extractedLoops.insert(loop);
  }
  return modified;
}

}  // namespace pass