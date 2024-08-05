#include "pass/optimize/Utils/BlockUtils.hpp"
#include "pass/optimize/Misc/StatelessCache.hpp"

using namespace ir;

namespace pass {
Function* StatelessCache::getLookupFunction(Module* module) {
  return nullptr;
}
bool StatelessCache::has2MoreRecursiveCalls(Function* func) {
  size_t count = 0;
  for (auto block : func->blocks()) {
    for (auto inst : block->insts()) {
      if (auto call = inst->dynCast<CallInst>()) {
        if (call->callee() == func) {
          count++;
        } else {
          return false;
        }
      }
    }  // for each instruction in block
  }  // for each block
  return count >= 2;
}

bool checkArgs(Function* func) {
  if (func->args().empty() or func->args().size() > 2) return false;
  for (auto argType : func->argTypes()) {
    if (not(argType->isInt32() or argType->isFloat32())) return false;
  }
  return true;
}

bool StatelessCache::runImpl(ir::Function* func, TopAnalysisInfoManager* tp) {
  // check if function is Stateless: NoMemoryRead, NoSideEffect
  // check if function has 2 more recursive calls
  if (not has2MoreRecursiveCalls(func)) return false;
  const auto i32 = Type::TypeInt32();
  const auto f32 = Type::TypeFloat32();
  const auto retType = func->retType();

  // only support i32 and f32 return types
  if (not(retType->isInt32() or retType->isFloat32())) return false;

  if (not checkArgs(func)) return false;

  /* split entry block to: alloca block, eval block */
  const auto entry = func->entry();
  const auto branchInst = entry->terminator()->as<BranchInst>();

  BasicBlock* evalBloack = nullptr;
  for (auto iter = entry->insts().begin(); iter != entry->insts().end(); iter++) {
    const auto inst = *iter;
    if (not inst->isa<AllocaInst>()) {
      evalBloack = splitBlock(func->blocks(), func->blocks().begin(), std::prev(iter));
      break;
    }
  }
  // fix phi inst: replace incoming block from entry to evalBlock
  std::vector<BasicBlock*> workList;
  if (branchInst->is_cond()) {
    workList.push_back(branchInst->iftrue());
    workList.push_back(branchInst->iffalse());
  } else {
    workList.push_back(branchInst->block());
  }
  for (auto block : workList) {
    for (auto inst : block->insts()) {
      if (auto phi = inst->dynCast<PhiInst>()) {
        phi->replaceoldtonew(entry, evalBloack);
      }
    }
  }

  IRBuilder builder;
  builder.set_pos(entry, entry->insts().end());

  // prepare lookup function, lookup table (lut)
  const size_t tableSize = 1021, tableWords = tableSize * 4;
  // totally tableSize lut entries, each entry is 4 words (i32)
  const auto lutType = ArrayType::gen(i32, {tableWords}, tableWords);

  const auto lut = utils::make<GlobalVariable>(lutType, "lut_" + func->name());
  // const auto lut = func->module()->addGlobalVar("lut_" + func->name(), lutType)
  // prepare lookup function arguments: (table, i32 key1, i32 key2)
  std::vector<Value*> lookupFuncArgs;
  lookupFuncArgs.push_back(lut);
  for (auto arg : func->args()) {
    if (arg->isInt32())
      lookupFuncArgs.push_back(arg);
    else if (arg->isFloat32())
      lookupFuncArgs.push_back(builder.makeInst<BitCastInst>(i32, arg));
  }
  while (lookupFuncArgs.size() < 3) {
    lookupFuncArgs.push_back(ConstantInteger::gen_i32(0));
  }
  const auto lookupFunc = getLookupFunction(func->module());

  // ptr = call lookup(table, key1, key2), return ptr is the pointer to the lookuped entry
  const auto entryPtr = builder.makeInst<CallInst>(lookupFunc, lookupFuncArgs);
  /*
  struct LUTEntry final {
      uint64_t key;
      int val;
      int hasVal;
  };
  */
  // load the value from the lookuped entry: *(ptr + 2) by words (i32), LUTEntry.val ptr
  auto valPtr = builder.makeGetElementPtr(i32, entryPtr, ConstantInteger::gen_i32(2), {}, {});
  if (not(valPtr->type()->as<PointerType>()->baseType() == retType)) {
    // cast val pointer to the return type pointer
    // valPtr = builder.makeInst<PtrCastInst>(valPtr, PointerType::gen(retType));
    valPtr = builder.makeInst<BitCastInst>(PointerType::gen(retType), valPtr);
  }
  // LUTEntry.hasVal ptr
  auto hasValPtr = builder.makeGetElementPtr(i32, entryPtr, ConstantInteger::gen_i32(3), {}, {});
  // load the hasVal from the lookuped entry: *(ptr + 3) by words (i32), LUTEntry.hasVal ptr
  auto hasVal = builder.makeLoad(hasValPtr);
  // cmp inst
  auto cmp = builder.makeCmp(CmpOp::NE, hasVal, ConstantInteger::gen_i32(0));
  // if true, jump to earlyExit block; else, jump to evalBlock
  auto earlyExitBlock = func->newBlock();
  earlyExitBlock->addComment("early exit block");

  builder.makeInst<BranchInst>(cmp, earlyExitBlock, evalBloack);

  // build entry end
  // earlyExitBlock: return the lookuped value directly
  builder.set_pos(earlyExitBlock);
  builder.makeInst<ReturnInst>(builder.makeLoad(valPtr));

  for (auto block : func->blocks()) {
    if (block == earlyExitBlock) continue;
    // not earlyExit, insert store hasVal and val
    // before return, store hasVal and val to the lookuped entry
    if (auto retInst = block->terminator()->dynCast<ReturnInst>()) {
      builder.set_pos(block, std::prev(block->insts().end()));
      // store hasVal to the lookuped entry
      builder.makeInst<StoreInst>(ConstantInteger::gen_i32(1), hasValPtr);
      // store val to the lookuped entry
      builder.makeInst<StoreInst>(retInst->returnValue(), valPtr);
    }
  }

  return true;
}

void StatelessCache::run(ir::Function* func, TopAnalysisInfoManager* tp) {
  runImpl(func, tp);
}
}  // namespace pass