#include "ir/function.hpp"
#include "ir/utils_ir.hpp"
#include "ir/type.hpp"
#include "ir/instructions.hpp"
#include "support/arena.hpp"
namespace ir {
BasicBlock* Function::newBlock() {
  auto nb = utils::make<BasicBlock>("", this);
  mBlocks.emplace_back(nb);
  return nb;
}

BasicBlock* Function::newEntry(const_str_ref name) {
  assert(mEntry == nullptr);
  mEntry = utils::make<BasicBlock>(name, this);
  mBlocks.emplace_back(mEntry);
  return mEntry;
}
BasicBlock* Function::newExit(const_str_ref name) {
  mExit = utils::make<BasicBlock>(name, this);
  mBlocks.emplace_back(mExit);
  return mExit;
}
void Function::delBlock(BasicBlock* bb) {
  for (auto bbpre : bb->pre_blocks()) {
    bbpre->next_blocks().remove(bb);
  }
  for (auto bbnext : bb->next_blocks()) {
    bbnext->pre_blocks().remove(bb);
  }
  for (auto bbinstIter = bb->insts().begin();
       bbinstIter != bb->insts().end();) {
    auto delinst = *bbinstIter;
    bbinstIter++;
    bb->delete_inst(delinst);
  }
  mBlocks.remove(bb);
  // delete bb;
}

void Function::forceDelBlock(BasicBlock* bb) {
  for (auto bbpre : bb->pre_blocks()) {
    bbpre->next_blocks().remove(bb);
  }
  for (auto bbnext : bb->next_blocks()) {
    bbnext->pre_blocks().remove(bb);
  }
  for (auto bbinstIter = bb->insts().begin();
       bbinstIter != bb->insts().end();) {
    auto delinst = *bbinstIter;
    bbinstIter++;
    bb->force_delete_inst(delinst);
  }
  mBlocks.remove(bb);
}

void Function::print(std::ostream& os) const {
  auto return_type = retType();
  if (blocks().size()) {
    os << "define " << *return_type << " @" << name() << "(";
    if (mArguments.size() > 0) {
      auto last_iter = mArguments.end() - 1;
      for (auto iter = mArguments.begin(); iter != last_iter; ++iter) {
        auto arg = *iter;
        os << *(arg->type()) << " " << arg->name();
        os << ", ";
      }
      auto arg = *last_iter;
      os << *(arg->type()) << " " << arg->name();
    }
  } else {
    os << "declare " << *return_type << " @" << name() << "(";
    auto t = type();
    if (auto TypeFunction = dyn_cast<FunctionType>(t)) {
      auto args_types = TypeFunction->argTypes();
      if (args_types.size() > 0) {
        auto last_iter = args_types.end() - 1;
        for (auto iter = args_types.begin(); iter != last_iter; ++iter) {
          os << **iter << ", ";
        }
        os << **last_iter;
      }
    } else {
      assert(false && "Unexpected type");
    }
  }

  os << ")";

  // print bbloks
  if (blocks().size()) {
    os << " {\n";
    for (auto& bb : mBlocks) {
      os << *bb << std::endl;
    }
    os << "}";
  } else {
    os << "\n";
  }
}

void Function::rename() {
  if (mBlocks.empty()) return;
  setVarCnt(0);
  for (auto arg : mArguments) {
    std::string argname = "%" + std::to_string(varInc());
    arg->set_name(argname);
  }
  size_t blockIdx = 0;
  for (auto bb : mBlocks) {
    bb->set_idx(blockIdx);
    blockIdx++;
    for (auto inst : bb->insts()) {
      if (inst->isNoName()) continue;
      auto callpt = dyn_cast<CallInst>(inst);
      if (callpt and callpt->isVoid()) continue;
      inst->setvarname();
    }
  }
}

// func_copy
Function* Function::copy_func() {
  std::unordered_map<Value*, Value*> ValueCopy;
  // copy global
  for (auto gvalue : mModule->globalVars()) {
    // if (dyn_cast<Constant>(gvalue) && !gvalue->type()->isPointer()) {
    //     ValueCopy[gvalue] = gvalue;  //??
    // } else {
    //     ValueCopy[gvalue] = gvalue;
    // }
    ValueCopy[gvalue] = gvalue;
  }
  // copy func
  auto copyfunc = utils::make<Function>(type(), name() + "_copy", mModule);
  // copy args
  for (auto arg : mArguments) {
    Value* copyarg = copyfunc->new_arg(arg->type(), "");
    ValueCopy[arg] = copyarg;
  }
  // copy block
  for (auto bb : mBlocks) {
    BasicBlock* copybb = copyfunc->newBlock();
    if (copyfunc->entry() == nullptr) {
      copyfunc->setEntry(copybb);
    } else if (copyfunc->exit() == nullptr) {
      copyfunc->setExit(copybb);
    }
    ValueCopy[bb] = copybb;
  }

  // copy bb's pred and succ
  for (auto BB : mBlocks) {
    auto copyBB = dyn_cast<BasicBlock>(ValueCopy[BB]);
    for (auto pred : BB->pre_blocks()) {
      copyBB->pre_blocks().emplace_back(dyn_cast<BasicBlock>(ValueCopy[pred]));
    }
    for (auto succ : BB->next_blocks()) {
      copyBB->next_blocks().emplace_back(dyn_cast<BasicBlock>(ValueCopy[succ]));
    }
  }

  auto getValue = [&](Value* val) -> Value* {
    if (auto c = dyn_cast<Constant>(val)) return c;
    return ValueCopy[val];
  };

  // copy inst in bb
  std::vector<PhiInst*> phis;
  std::set<BasicBlock*> vis;
  BasicBlock::BasicBlockDfs(mEntry, [&](BasicBlock* bb) -> bool {
    if (vis.count(bb)) return true;
    vis.insert(bb);
    auto bbCpy = dyn_cast<BasicBlock>(ValueCopy[bb]);
    for (auto inst : bb->insts()) {
      auto copyinst = inst->copy_inst(getValue);
      copyinst->setBlock(bbCpy);
      ValueCopy[inst] = copyinst;
      bbCpy->emplace_back_inst(copyinst);
      if (auto phi = dyn_cast<PhiInst>(inst)) phis.emplace_back(phi);
    }
    return false;
  });
  for (auto phi : phis) {
    auto copyphi = dyn_cast<PhiInst>(ValueCopy[phi]);
    for (size_t i = 0; i < phi->getsize(); i++) {
      auto phivalue = getValue(phi->getValue(i));
      auto phibb = dyn_cast<BasicBlock>(getValue(phi->getBlock(i)));
      copyphi->addIncoming(phivalue, phibb);
    }
  }
  return copyfunc;
}

bool Function::verify(std::ostream& os) const {
  for (auto block : mBlocks) {
    if (not block->verify(os)) {
      return false;
    }
  }
  return true;
}

}  // namespace ir