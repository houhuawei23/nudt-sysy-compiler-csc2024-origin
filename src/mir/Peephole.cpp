#include <queue>
#include "mir/mir.hpp"
#include "mir/instinfo.hpp"
#include "mir/utils.hpp"

namespace mir {
/*
 * @brief: eliminateStackLoads function
 * @note: 
 *     消除无用的Load Register from Stack指令
 *     使用前面已经load之后的虚拟寄存器Copy指令来代替   
 */
bool eliminateStackLoads(MIRFunction& mfunc, CodeGenContext& ctx) {
    bool modified = false;
    // for (auto& block : mfunc.blocks()) {
    //     auto& instructions = block->insts();

    //     uint32_t versionId = 0;
    //     std::unordered_map<RegNum, uint32_t> reg2Version;
    //     auto defReg = [&](RegNum reg) { reg2Version[reg] = ++versionId; };
    //     std::unordered_map<MIROperand, std::pair<RegNum, uint32_t>, MIROperandHasher> stack2Reg;  // stack -> (reg, version)

    //     for (auto inst : instructions) {
    //         if (inst->opcode() == InstStoreRegToStack) {
    //             auto obj = inst->operand(0);
    //             auto reg = inst->operand(1);
    //             if (auto iter = reg2Version.find(reg.reg()); iter != reg2Version.cend()) {
    //                 stack2Reg[obj] = { reg.reg(), iter->second };
    //             } else {
    //                 defReg(reg->reg());
    //                 stack2Reg[obj] = { reg->reg(), versionId };
    //             }
    //         } else if (inst->opcode() == InstLoadRegFromStack) {
    //             auto dst = inst->operand(0);
    //             auto obj = inst->operand(1);
    //             if (auto iter = stack2Reg.find(obj); iter != stack2Reg.cend()) {
    //                 auto& [reg, ver] = stack2Reg[obj];
    //                 if (ver == reg2Version[reg]) {
    //                     // dst <- reg
    //                     inst->set_opcode(InstCopy);
    //                     if (isVirtualReg(reg)) {
    //                         auto vReg = MIROperand::asVReg(reg, dst->type());
    //                         *obj = *vReg;
    //                         delete vReg;
    //                     } else if (isISAReg(reg)) {
    //                         auto phyReg = MIROperand::asISAReg(reg, dst->type());
    //                         *obj = *phyReg;
    //                         delete phyReg;
    //                     } else {
    //                         assert(false && "not supported type");
    //                     }
    //                     modified = true;
    //                 }
    //             }
    //         }

    //         auto& instInfo = ctx.instInfo.get_instinfo(inst);
    //         for (uint32_t idx = 0; idx < instInfo.operand_num(); idx++) {
    //             if (instInfo.operand_flag(idx) & OperandFlagDef) {
    //                 defReg(inst->operand(idx)->reg());
    //             }
    //         }

    //         if (requireFlag(instInfo.inst_flag(), InstFlagCall)) {
    //             std::vector<RegNum> nonVReg;
    //             for (auto [reg, ver] : reg2Version) {
    //                 // TODO: use IPRA Info
    //                 if (isISAReg(reg)) nonVReg.push_back(reg);
    //             }
    //             for (auto reg : nonVReg) defReg(reg);
    //         }

    //         if (inst->opcode() == InstLoadRegFromStack) {
    //             auto dst = inst->operand(0);
    //             auto obj = inst->operand(1);
    //             stack2Reg[obj] = { dst->reg(), reg2Version.at(dst->reg()) };
    //         }
    //     }
    // }
    
    return modified;
}
/*
 * @brief: removeIndirectCopy
 * @note: TODO
 */
bool removeIndirectCopy(MIRFunction& mfunc, CodeGenContext& ctx) {
    if (ctx.flags.dontForward) return false;  // ???

    constexpr bool Debug = true;
    if (Debug) {
        std::cerr << "before optimize: \n";
        mfunc.print(std::cerr, ctx);
    }
    bool modified = false;

    // for(auto& block : mfunc.blocks()) {
    //     auto& instructions = block->insts();

    //     uint32_t versionId = 0;
    //     std::unordered_map<RegNum, std::pair<RegNum, uint32_t>> reg2Value;  // reg -> (reg, version)
    //     std::unordered_map<RegNum, uint32_t> version;  // reg -> version
    //     const auto getVersion = [&](const uint32_t reg) {
    //         assert(isVirtualReg(reg) || isISAReg(reg));
    //         if(auto iter = version.find(reg); iter != version.cend()) return iter->second;
    //         return version[reg] = ++versionId;
    //     };
    //     const auto defReg = [&](MIROperand* reg) {
    //         if(!isOperandVRegOrISAReg(reg)) return;
    //         version[reg->reg()] = ++versionId;
    //         reg2Value.erase(reg->reg());
    //     };

    //     const auto replaceUse = [&](MIRInst* inst, MIROperand* reg) {
    //         if(!isOperandVRegOrISAReg(reg)) return;
    //         if(auto iter = reg2Value.find(reg->reg()); iter != reg2Value.cend() && iter->second.second == getVersion(iter->second.first)) {
    //             if(ctx.flags.preRA && (!isVirtualReg(iter->second.first) &&
    //                !(ctx.registerInfo && ctx.registerInfo->is_zero_reg(iter->second.first))))
    //                return;  // should be handled after RA
    //             auto backup = reg;
    //             // NOTICE: Don't modify the type
    //             reg = MIROperand{ MIRRegister{ iter->second.first }, backup.type() };
    //             if(reg == backup)
    //                 return;
    //             auto backupInstOpcode = inst.opcode();
    //             if(inst.opcode() == InstCopy) {
    //                 inst.setOpcode(selectCopyOpcode(inst.getOperand(0), reg));
    //             }
    //             auto& instInfo = ctx.instInfo.getInstInfo(inst);
    //             if(instInfo.verify(inst, ctx)) {
    //                 modified = true;
    //             } else {
    //                 reg = backup;
    //                 inst.setOpcode(backupInstOpcode);
    //             }
    //         }
    //     };

    //     for(auto iter = instructions.begin(); iter != instructions.end();) {
    //         auto& inst = *iter;
    //         auto next = std::next(iter);

    //         auto& instInfo = ctx.instInfo.getInstInfo(inst);
    //         for(uint32_t idx = 0; idx < instInfo.getOperandNum(); ++idx)
    //             if(instInfo.getOperandFlag(idx) & OperandFlagUse) {
    //                 auto& operand = inst.getOperand(idx);
    //                 replaceUse(inst, operand);
    //             }

    //         MIROperand dst, src;
    //         if(ctx.instInfo.matchCopy(inst, dst, src)) {
    //             assert(isOperandVRegOrISAReg(dst) && isOperandVRegOrISAReg(src));
    //             if(auto it = regValue.find(dst.reg());
    //                it != regValue.cend() && it->second.first == src.reg() && it->second.second == getVersion(it->second.first)) {
    //                 instructions.erase(iter);
    //                 modified = true;
    //             } else {
    //                 defReg(dst);
    //                 regValue[dst.reg()] = { src.reg(), getVersion(src.reg()) };
    //             }
    //         } else {
    //             for(uint32_t idx = 0; idx < instInfo.getOperandNum(); ++idx)
    //                 if(instInfo.getOperandFlag(idx) & OperandFlagDef) {
    //                     defReg(inst.getOperand(idx));
    //                 }
    //             if(requireFlag(instInfo.getInstFlag(), InstFlagCall)) {
    //                 std::vector<uint32_t> nonVReg;
    //                 for(auto [reg, ver] : version) {
    //                     CMMC_UNUSED(ver);
    //                     // TODO: use IPRA Info
    //                     if(isISAReg(reg))
    //                         nonVReg.push_back(reg);
    //                 }
    //                 for(auto reg : nonVReg) {
    //                     version[reg] = ++versionId;
    //                     regValue.erase(reg);
    //                 }
    //             }
    //         }

    //         iter = next;
    //     }
    // }

    // if (Debug) {
    //     std::cerr << "after optimize: \n";
    //     mfunc.print(std::cerr, ctx); 
    // }

    return modified;
}
bool removeIdentityCopies(MIRFunction& func, CodeGenContext& ctx) {
  bool modified = false;
  for (auto& block : func.blocks()) {
    block->insts().remove_if([&](MIRInst* inst) {
      MIROperand dst, src;
      ctx.instInfo.matchCopy(inst, dst, src);
      auto& info = ctx.instInfo.getInstInfo(inst);
      if (dst == src) {
        std::cerr << "remove identity copy: ";
        info.print(std::cerr, *inst, false);
        std::cerr << std::endl;
        modified = true;
        return true;
      }
      return false;
    });
  }
  return modified;
}
bool removeUnusedInsts(MIRFunction& func, CodeGenContext& ctx) {
    /* writers: map from operand to list of insts that write it */
    std::unordered_map<MIROperand, std::vector<MIRInst*>, MIROperandHasher> writers;

    /** specail insts: that cant be removed,
     * 1 InstFlagSideEffect,
     * 2 def reg is allocable type */
    std::queue<MIRInst*> q;

    auto isAllocableType = [](OperandType type) {
        return type <= OperandType::Float32;
    };

    for (auto& block : func.blocks()) {
        for (auto& inst : block->insts()) {
            auto& instInfo = ctx.instInfo.getInstInfo(inst);
            bool special = false;

            if (requireOneFlag(instInfo.inst_flag(), InstFlagSideEffect)) {
                special = true;
            }

            for (uint32_t idx = 0; idx < instInfo.operand_num(); idx++) {
                if (instInfo.operand_flag(idx) & OperandFlagDef) {
                    auto op = inst->operand(idx);
                    writers[op].push_back(inst);
                    if (op.isReg() && isISAReg(op.reg()) &&
                        isAllocableType(op.type())) {
                        special = true;
                    }
                }
            }
            if (special) {
                q.push(inst);
            }
        }
    }

    /*
    i1: sub b[Def], 0[Metadata], 1[Metadata]
    i2: add b[Def], 1[Metadata], 2[Metadata]
    i3: store a[Def], b[Use]

    q0 = {i1, i2, i3}
    writers: {
        a: {i3},
        b: {i1, i2},
    }

    inst = i1, pop i1, q = {i2, i3}
    writers not changed

    inst = i2, pop i2, q = {i3}
    writers not changed

    inst = i3, pop i3, q = {}
    writers[b] = {i1, i2}, q.push(i1, i2), q = {i1, i2}, writers erase b
    writers = { a: {i3} }

    inst = i1, pop i1, q = {i2}
    writers not changed

    inst = i2, pop i2, q = {}
    writers not changed

    finieshed, writers = { a: {i3} }

    */
    /*
    q has insts that cant be removed and not yet processed,
    if inst i cant be removed, then the insts define i's operand also cant be
    removed, so remove i's operand from writers, and add insts that define i's
    operand to q.
    */
    while (not q.empty()) {
        auto& inst = q.front();
        q.pop();

        auto& instInfo = ctx.instInfo.getInstInfo(inst);
        for (uint32_t idx = 0; idx < instInfo.operand_num(); idx++) {
            if (instInfo.operand_flag(idx) & OperandFlagUse) {
                /* if operand is used, remove from writers */
                auto op = inst->operand(idx);
                if (auto iter = writers.find(op); iter != writers.end()) {
                    for (auto& writer : iter->second) {
                        q.push(writer);
                    }
                    writers.erase(iter);
                }
            }
        }
    }
    /* after while, writers contain operands that are not used */

    std::unordered_set<MIRInst*> remove;
    for (auto& [op, writerList] : writers) {
        if (isISAReg(op.reg()) && isAllocableType(op.type())) {
            continue;
        }
        for (auto& writer : writerList) {
            auto& instInfo = ctx.instInfo.getInstInfo(writer);
            if (requireOneFlag(instInfo.inst_flag(),
                               InstFlagSideEffect | InstFlagMultiDef)) {
                continue;
            }
            remove.insert(writer);
        }
    }

    for (auto& block : func.blocks()) {
        block->insts().remove_if(
            [&](MIRInst* inst) { return remove.count(inst); });
    }

    if(not remove.empty()) {
        int a = 12;
        int b = remove.size();
        std::cout << "removed " << b << " insts" << std::endl;
    }

    return not remove.empty();
}
bool applySSAPropagation(MIRFunction& func, CodeGenContext& ctx) {
    return false;
}
bool machineConstantCSE(MIRFunction& func, CodeGenContext& ctx) {
    return false;
}
bool machineConstantHoist(MIRFunction& func, CodeGenContext& ctx) {
    return false;
}
bool machineInstCSE(MIRFunction& func, CodeGenContext& ctx) {
    return false;
}
bool deadInstElimination(MIRFunction& func, CodeGenContext& ctx) {
    bool modified = false;
    for (auto& block : func.blocks()) {
        auto& instructions = block->insts();
    }
    
    return modified;
}
bool removeInvisibleInsts(MIRFunction& func, CodeGenContext& ctx) {
    return false;
}
bool genericPeepholeOpt(MIRFunction& func, CodeGenContext& ctx) {
    return false;
    bool modified = false;
    modified |= eliminateStackLoads(func, ctx);
    // modified |= removeIndirectCopy(func, ctx);
    // modified |= removeIdentityCopies(func, ctx);
    // // modified |= removeUnusedInsts(func, ctx);
    // modified |= applySSAPropagation(func, ctx);
    // modified |= machineConstantCSE(func, ctx);
    // modified |= machineConstantHoist(func, ctx);
    // modified |= machineInstCSE(func, ctx);
    // modified |= deadInstElimination(func, ctx);
    // modified |= removeInvisibleInsts(func, ctx);
    // modified |= ctx.scheduleModel->peepholeOpt(func, ctx);
    return modified;
}
}