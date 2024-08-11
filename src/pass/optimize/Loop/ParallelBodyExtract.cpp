#include "pass/optimize/Loop/ParallelBodyExtract.hpp"
#include "pass/optimize/Loop/LoopBodyExtract.hpp"
#include "pass/analysis/ControlFlowGraph.hpp"
#include "pass/optimize/Utils/BlockUtils.hpp"
using namespace ir;

namespace pass {

bool extractParallelBody(Function* func,
                         Loop* loop,
                         IndVar* indVar,
                         TopAnalysisInfoManager* tp,
                         ParallelBodyInfo& parallelBodyInfo) {
  const auto step = indVar->getStep()->i32();
  indVar->print(std::cerr);

  if (step != 1) return false;                  // only support step = 1
  if (loop->exits().size() != 1) return false;  // only support single exit loop
  const auto loopExitBlock = *(loop->exits().begin());
  // extact loop body as a new loop_body func from func.loop
  /**
   * before:
   * entry -> body/header -> latch -> exit
   * after:
   * entry -> newLoop{phi, call loop_body, i.next, icmp, br} -> exit
   */
  LoopBodyFuncInfo loopBodyInfo;
  if (not extractLoopBody(func, *loop /* modified */, indVar, tp, loopBodyInfo /* ret */))
    return false;

  loopBodyInfo.print(std::cerr);
  const auto i32 = Type::TypeInt32();
  auto funcType = FunctionType::gen(Type::void_type(), {i32, i32});
  auto parallelBody = func->module()->addFunction(funcType, "parallel_body");

  auto argBeg = parallelBody->new_arg(i32, "beg");
  auto argEnd = parallelBody->new_arg(i32, "end");

  // update value with args
  // auto beginVar =
  // std::unordered_map<Value*, Value*> valueMap;
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
    std::cerr << "fix phi inst: ";
    phi->dumpAsOpernd(std::cerr);
    std::cerr << std::endl;
    if (phi == indVar->phiinst()) {
      phi->delBlock(loopBodyInfo.preHeader);
      phi->addIncoming(argBeg, newEntry);
      return;
    }
    std::cerr << "phi inst not indvar phi inst" << std::endl;
    phi->print(std::cerr);
    std::cerr << std::endl;
  };

  const auto fixCmp = [&](ICmpInst* cmpInst) {
    if (cmpInst == indVar->cmpInst()) {
      for (auto opuse : cmpInst->operands()) {
        auto op = opuse->value();
      }
    }
    std::cerr << "cmp inst not indvar cmp inst" << std::endl;
    cmpInst->print(std::cerr);
    std::cerr << std::endl;
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
    std::cerr << "block: " << block->name() << std::endl;
    for (auto inst : block->insts()) {
      inst->print(std::cerr);
      std::cerr << std::endl;
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
    function->print(std::cerr);
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
void ParallelBodyExtract::runImpl(ir::Function* func, TopAnalysisInfoManager* tp) {
  func->rename();
  func->print(std::cerr);

  CFGAnalysisHHW().run(func, tp);  // refresh CFG

  auto lpctx = tp->getLoopInfo(func);         // fisrt loop analysis
  auto indVarInfo = tp->getIndVarInfo(func);  // then indvar analysis

  // for all loops
  for (auto loop : lpctx->loops()) {
    loop->print(std::cerr);
    const auto indVar = indVarInfo->getIndvar(loop);
    ParallelBodyInfo info;
    if (not extractParallelBody(func, loop, indVar, tp, info)) {
      std::cerr << "failed to extract parallel body for loop" << std::endl;
    }
  }
}

}  // namespace pass