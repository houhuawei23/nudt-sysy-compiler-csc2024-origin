
#include "mir/MIR.hpp"
#include "mir/target.hpp"
#include "mir/LiveInterval.hpp"
#include "mir/RegisterAllocator.hpp"
#include <queue>
#include <unordered_set>
#include <iostream>

namespace mir {
void fastAllocatorBeta(MIRFunction& mfunc,
                              CodeGenContext& ctx,
                              IPRAUsageCache& infoIPRA) {
  auto liveInterval = calcLiveIntervals(mfunc, ctx);
  // mfunc.print(std::cerr, ctx);

  struct VirtualRegUseInfo final {
    std::unordered_set<MIRBlock*> uses;
    std::unordered_set<MIRBlock*> defs;
  };
  std::unordered_map<MIROperand, VirtualRegUseInfo, MIROperandHasher>
    useDefInfo;
  std::unordered_map<MIROperand, MIROperand, MIROperandHasher> isaRegHint;

  for (auto& block : mfunc.blocks()) {
    for (auto& inst : block->insts()) {
      auto& instInfo = ctx.instInfo.getInstInfo(inst);

      if (inst->opcode() == InstCopyFromReg) {
        /* CopyFromReg $Dst:VRegOrISAReg[Def], $Src:ISAReg[Use] */
        isaRegHint[inst->operand(0)] = inst->operand(1);
      }
      if (inst->opcode() == InstCopyToReg) {
        isaRegHint[inst->operand(1)] = inst->operand(0);
      }

      for (uint32_t idx = 0; idx < instInfo.operand_num(); ++idx) {
        const auto& op = inst->operand(idx);
        if (!isOperandVReg(op)) {
          continue;
        }
        if (instInfo.operand_flag(idx) & OperandFlagUse) {
          useDefInfo[op].uses.insert(block.get());
        }
        if (instInfo.operand_flag(idx) & OperandFlagDef) {
          useDefInfo[op].defs.insert(block.get());
        }
      }
    }
  }

  // find all cross-block vregs and allocate stack slots for them
  std::unordered_map<MIROperand, MIROperand, MIROperandHasher> stackMap;
  std::unordered_map<MIROperand, MIROperand, MIROperandHasher> isaRegStackMap;

  {
    for (auto& [reg, info] : useDefInfo) {
      if (info.uses.empty() || info.defs.empty()) {
        continue;  // invalid
      }
      if (info.uses.size() == 1 && info.defs.size() == 1 &&
          *info.uses.cbegin() == *info.defs.cbegin()) {
        continue;  // local
      }

      const auto size = getOperandSize(
        ctx.registerInfo->getCanonicalizedRegisterType(reg.type()));
      const auto storage = mfunc.newStackObject(ctx.nextId(), size, size, 0,
                                                StackObjectUsage::RegSpill);
      stackMap[reg] = storage;
    }
  }

  for (auto& block : mfunc.blocks()) {
    std::unordered_map<MIROperand, MIROperand, MIROperandHasher> localStackMap;
    std::unordered_map<MIROperand, std::vector<MIROperand>, MIROperandHasher>
      currentMap;
    std::unordered_map<MIROperand, MIROperand, MIROperandHasher> physMap;
    std::unordered_map<uint32_t, std::queue<MIROperand>> allocationQueue;
    std::unordered_set<MIROperand, MIROperandHasher>
      protectedLockedISAReg;  // retvals/callee arguments
    std::unordered_set<MIROperand, MIROperandHasher>
      underRenamedISAReg;  // callee retvals/arguments

    MultiClassRegisterSelector selector{*ctx.registerInfo};

    const auto getStackStorage = [&](const MIROperand& op) {
      if (const auto iter = localStackMap.find(op);
          iter != localStackMap.cend()) {
        return iter->second;
      }
      auto& ref = localStackMap[op];
      if (const auto iter = stackMap.find(op); iter != stackMap.cend()) {
        return ref = iter->second;
      }
      const auto size = getOperandSize(
        ctx.registerInfo->getCanonicalizedRegisterType(op.type()));
      const auto storage = mfunc.newStackObject(ctx.nextId(), size, size, 0,
                                                StackObjectUsage::RegSpill);
      return ref = storage;
    };
    const auto getDataMap =
      [&](const MIROperand& op) -> std::vector<MIROperand>& {
      auto& map = currentMap[op];
      if (map.empty()) map.push_back(getStackStorage(op));
      return map;
    };

    auto& insts = block->insts();

    std::unordered_set<MIROperand, MIROperandHasher> dirtyVRegs;
    auto& liveIntervalInfo = liveInterval.block2Info[block.get()];

    const auto isAllocatableType = [](OperandType type) {
      return type <= OperandType::Float32;
    };
    const auto collectUnderRenamedISARegs = [&](MIRInstList::iterator it) {
      while (it != insts.end()) {
        const auto& inst = *it;
        auto& instInfo = ctx.instInfo.getInstInfo(inst);
        bool hasReg = false;
        for (uint32_t idx = 0; idx < instInfo.operand_num(); ++idx) {
          const auto& op = inst->operand(idx);
          if (isOperandISAReg(op) && !ctx.registerInfo->is_zero_reg(op.reg()) &&
              isAllocatableType(op.type()) &&
              (instInfo.operand_flag(idx) & OperandFlagUse)) {
            underRenamedISAReg.insert(op);
            hasReg = true;
          }
        }
        if (hasReg)
          ++it;
        else
          break;
      }
    };

    collectUnderRenamedISARegs(insts.begin());

    for (auto iter = insts.begin(); iter != insts.end();) {
      const auto next = std::next(iter);

      const auto evictVReg = [&](MIROperand operand) {
        assert(isOperandVReg(operand));
        auto& map = getDataMap(operand);
        MIROperand isaReg;
        bool alreadyInStack = false;
        for (auto& reg : map) {
          if (isStackObject(reg.reg())) {
            alreadyInStack = true;
          }
          if (isISAReg(reg.reg())) isaReg = reg;
        }
        if (isaReg.isUnused()) return;
        physMap.erase(isaReg);
        const auto stackStorage = getStackStorage(operand);
        if (!alreadyInStack) {
          // spill to stack
          // insts.insert(
          //   iter,
          //   MIRInst{InstStoreRegToStack}.setOperand<0>(isaReg).setOperand<1>(
          //     stackStorage));
          auto inst = new MIRInst(InstStoreRegToStack);
          inst->set_operand(0, MIROperand(stackStorage));
          inst->set_operand(
            1, MIROperand::asISAReg(
                 isaReg.reg(),
                 ctx.registerInfo->getCanonicalizedRegisterType(isaReg.reg())));
          insts.insert(iter, inst);
        }
        map = {stackStorage};
      };

      std::unordered_set<MIROperand, MIROperandHasher> protect;
      const auto isProtected = [&](const MIROperand& isaReg) {
        assert(isOperandISAReg(isaReg));
        return protect.count(isaReg) || protectedLockedISAReg.count(isaReg) ||
               underRenamedISAReg.count(isaReg);
      };
      const auto getFreeReg = [&](const MIROperand& operand) -> MIROperand {
        const auto regClass =
          ctx.registerInfo->getAllocationClass(operand.type());
        auto& q = allocationQueue[regClass];
        MIROperand isaReg;

        const auto getFreeRegister = [&] {
          std::vector<MIROperand> temp;
          do {
            auto reg = selector.getFreeRegister(operand.type());
            if (reg.isUnused()) {
              for (auto op : temp)
                selector.markAsDiscarded(op);
              return MIROperand{};
            }
            if (isProtected(reg)) {
              temp.push_back(reg);
              selector.markAsUsed(reg);
            } else {
              for (auto op : temp)
                selector.markAsDiscarded(op);
              return reg;
            }
          } while (true);
        };

        if (auto hintIter = isaRegHint.find(operand);
            hintIter != isaRegHint.end() && selector.isFree(hintIter->second) &&
            !isProtected(hintIter->second)) {
          isaReg = hintIter->second;
        } else if (auto reg = getFreeRegister(); !reg.isUnused()) {
          isaReg = reg;
        } else {
          // evict
          assert(!q.empty());
          isaReg = q.front();
          while (isProtected(isaReg)) {
            assert(q.size() != 1);
            q.pop();
            q.push(isaReg);
            isaReg = q.front();
          }
          q.pop();
          selector.markAsDiscarded(isaReg);
        }
        if (auto it = physMap.find(isaReg); it != physMap.cend())
          evictVReg(it->second);
        assert(!isProtected(isaReg));

        // std::cerr << (operand.reg() - virtualRegBegin) << " -> " <<
        // isaReg.reg() << std::endl;

        q.push(isaReg);
        physMap[isaReg] = operand;
        selector.markAsUsed(isaReg);
        return isaReg;
      };

      std::unordered_set<MIROperand, MIROperandHasher> releaseVRegs;

      const auto use = [&](MIROperand& op) {
        if (!isOperandVReg(op)) {
          if (isOperandISAReg(op) &&
              !ctx.registerInfo->is_zero_reg(op.reg()) &&
              isAllocatableType(op.type())) {
            underRenamedISAReg.erase(op);
          }
          return;
        }
        if (op.reg_flag() & RegisterFlagDead) releaseVRegs.insert(op);

        auto& map = getDataMap(op);
        MIROperand stackStorage;
        for (auto& reg : map) {
          if (!isStackObject(reg.reg())) {
            // loaded
            op = reg;
            protect.insert(reg);
            return;
          }
          stackStorage = reg;
        }
        // load from stack
        assert(!stackStorage.isUnused());
        const auto reg = getFreeReg(op);
        // insts.insert(
        //   iter, MIRInst{InstLoadRegFromStack}.setOperand<0>(reg).setOperand<1>(
        //           stackStorage));
        auto inst = new MIRInst(InstLoadRegFromStack);
        inst->set_operand(0, MIROperand(reg));
        inst->set_operand(1, MIROperand(stackStorage));
        insts.insert(iter, inst);
        map.push_back(reg);
        op = reg;
        protect.insert(reg);
      };

      const auto def = [&](MIROperand& op) {
        if (!isOperandVReg(op)) {
          if (isOperandISAReg(op) &&
              !ctx.registerInfo->is_zero_reg(op.reg()) &&
              isAllocatableType(op.type())) {
            protectedLockedISAReg.insert(op);
            if (auto it = physMap.find(op); it != physMap.cend())
              evictVReg(it->second);
          }
          return;
        }

        if (stackMap.count(op)) {
          dirtyVRegs.insert(op);
        }

        auto& map = getDataMap(op);
        MIROperand stackStorage;
        for (auto& reg : map) {
          if (!isStackObject(reg.reg())) {
            op = reg;
            map = {reg};  // mark other storage dirty
            protect.insert(reg);
            return;
          }
          stackStorage = reg;
        }
        const auto reg = getFreeReg(op);
        map = {reg};
        protect.insert(reg);
        op = reg;
      };

      const auto beforeBranch = [&]() {
        // write back all out dirty vregs into stack slots before branch
        for (auto dirty : dirtyVRegs) {
          if (liveIntervalInfo.outs.count(dirty.reg())) evictVReg(dirty);
        }
      };

      auto& inst = *iter;
      auto& instInfo = ctx.instInfo.getInstInfo(inst);
      // instInfo.print(std::cerr, inst, true);
      // std::cerr << '\n';
      for (uint32_t idx = 0; idx < instInfo.operand_num(); ++idx) {
        if (auto flag = instInfo.operand_flag(idx);
            (flag & OperandFlagUse) || (flag & OperandFlagDef)) {
          const auto& op = inst->operand(idx);
          if (!isOperandVReg(op)) {
            if (isOperandISAReg(op)) protect.insert(op);
          }
        }
      }
      for (uint32_t idx = 0; idx < instInfo.operand_num(); ++idx) {
        if (instInfo.operand_flag(idx) & OperandFlagUse)
          use(inst->operand(idx));
      }
      if (requireFlag(instInfo.inst_flag(), InstFlagCall)) {
        std::vector<MIROperand> savedVRegs;
        const IPRAInfo* calleeUsage = nullptr;
        if (auto symbol = inst->operand(0).reloc()) {
          calleeUsage = infoIPRA.query(symbol->name());
        }
        for (auto& [p, v] : physMap) {
          if (ctx.frameInfo.isCallerSaved(p)) {
            if (calleeUsage && !calleeUsage->count(p.reg())) continue;
            savedVRegs.push_back(v);
          }
        }
        for (auto v : savedVRegs)
          evictVReg(v);
        protectedLockedISAReg.clear();
        collectUnderRenamedISARegs(next);
      }
      protect.clear();
      for (auto operand : releaseVRegs) {
        // release dead vregs
        auto& map = getDataMap(operand);
        for (auto& reg : map) {
          if (isISAReg(reg.reg())) {
            physMap.erase(reg);
            selector.markAsDiscarded(reg);
          }
        }
        map.clear();
      }

      for (uint32_t idx = 0; idx < instInfo.operand_num(); ++idx) {
        if (instInfo.operand_flag(idx) & OperandFlagDef)
          def(inst->operand(idx));
      }
      if (requireFlag(instInfo.inst_flag(), InstFlagBranch)) {
        beforeBranch();
      }

      iter = next;
    }

    assert(block->verify(std::cerr, ctx));
  }

  // mfunc.print(std::cerr, ctx);
}

}  // namespace mir
