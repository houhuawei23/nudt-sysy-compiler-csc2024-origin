#include "ir/ir.hpp"
#include "mir/mir.hpp"
#include "mir/target.hpp"
#include "mir/instinfo.hpp"
#include "mir/iselinfo.hpp"
#include "mir/utils.hpp"
#include <optional>

namespace mir {
auto collect_def_count(
    MIRFunction* func,
    CodeGenContext& codegen_ctx) {
    std::cerr << "collect_def_count not implemented yet!" << std::endl;
    return std::unordered_map<MIROperand, uint32_t, MIROperandHasher>();
}

void ISelContext::remove_inst(MIRInst* inst) {
    assert(inst != nullptr);
    mRemoveWorkList.insert(inst);
}
void ISelContext::replace_operand(MIROperand src, MIROperand dst) {
    assert(src.isReg());
    if (src != dst) {
        mReplaceMap.emplace(src, dst);
    }
}
MIROperand ISelContext::get_inst_def(MIRInst* inst) {
    assert(inst != nullptr);
    auto& instinfo = mCodeGenCtx.instInfo.getInstInfo(inst);
    for (uint32_t idx = 0; idx < instinfo.operand_num(); idx++) {
        if (instinfo.operand_flag(idx) & OperandFlagDef) {
            return inst->operand(idx);
        }
    }
    // assert(false && "no def operand found");
    std::cerr << "no def operand found" << std::endl;
    // return nullptr;
}

MIRInst* ISelContext::lookup_def(MIROperand op) {
    // assert(op != nullptr);
    auto iter = mInstMap.find(op.reg());
    if (iter != mInstMap.end()) {
        return iter->second;
    }

    // std::cerr << "op address " << op << "\n";
    if (isOperandVReg(op)) {
        std::cerr << "virtual reg v" << (op.reg() ^ virtualRegBegin) << "\n";
    } else if (isOperandISAReg(op)) {
        std::cerr << "physical reg i" << op.reg() << "\n";
    } else {
        std::cerr << "satck\n";
    }

    assert(false && "def not found");
}

void ISelContext::run_isel(MIRFunction* func) {
    bool debugISel = false;
    auto dumpInst = [&](MIRInst* inst) {
        auto& instInfo = mCodeGenCtx.instInfo.getInstInfo(inst);
        instInfo.print(std::cerr << "match&select: ", *inst, false);
        std::cerr << std::endl;
    };

    auto& isel_info = mCodeGenCtx.iselInfo;

    //! fix point algorithm: 循环执行指令选择和替换，直到不再变化。
    while (true) {
        genericPeepholeOpt(*func, mCodeGenCtx);

        bool modified = false;
        bool has_illegal = false;
        mRemoveWorkList.clear();
        mReplaceList.clear();
        mReplaceMap.clear();

        mConstantMap.clear();
        mUseCnt.clear();
        mInstMap.clear();  //! here!
        //! 定义和使用计数收集: 遍历所有指令，收集每个定义的计数和使用情况。
        // get def count
        // auto def_count = collect_def_count(func, _codegen_ctx);
        // for (auto& block : func->blocks()) {
        //     for (auto& inst : block->insts()) {
        //         // for all insts
        //         auto& instinfo = _codegen_ctx.instInfo.get_instinfo(inst);
        //         if (requireFlag(instinfo.inst_flag(), InstFlagLoadConstant))
        //         {
        //             // load constant, and def once, can view as constant
        //             // auto& def =
        //         }
        //     }
        // }

        //! 指令遍历和分析: 对每个基本块的指令进行遍历，执行指令选择和替换。
        // std::cerr << "function " << func->name() << "\n";
        for (auto& block : func->blocks()) {
            if (debugISel) {
                std::cout << block->name() << std::endl;
            }

            // check ssa form, get inst map
            for (auto& inst : block->insts()) {
                auto& instinfo = mCodeGenCtx.instInfo.getInstInfo(inst);
                for (uint32_t idx = 0; idx < instinfo.operand_num(); idx++) {
                    if (instinfo.operand_flag(idx) & OperandFlagDef) {
                        auto def = inst->operand(idx);
                        if (def.isReg() && isVirtualReg(def.reg())) {
                            // std::cerr << "def reg v " << (def.reg() ^ virtualRegBegin) << "\n";
                            // std::cerr << "def address " << def << "\n";
                            mInstMap.emplace(def.reg(), inst);
                        }
                    }
                }
            }
            mCurrBlock = block.get();
            auto& insts = block->insts();

            if (insts.empty())
                continue;

            auto it = std::prev(insts.end());
            /* in this block */
            while (true) {  // for all insts in block
                mInsertPoint = it;
                auto& inst = *it;
                std::optional<std::list<MIRInst*>::iterator> prev;

                if (it != insts.begin()) {
                    prev = std::prev(it);
                }
                // if inst not in remove list
                if (not mRemoveWorkList.count(inst)) {
                    if (debugISel) {
                        dumpInst(inst);
                    }

                    auto opcode = inst->opcode();
                    //! do pattern match and select inst
                    auto res = isel_info->match_select(inst, *this);
                    if (res) {
                        modified = true;
                    }
                }

                if (prev) {
                    it = *prev;
                } else {
                    break;
                }
            }
        }

        //! 指令移除和替换: 根据之前的分析结果，移除和替换旧的指令。
        for (auto& block : func->blocks()) {
            // remove old insts
            block->insts().remove_if(
                [&](auto inst) { return mRemoveWorkList.count(inst); });

            // replace defs
            for (auto& inst : block->insts()) {
                if (mReplaceList.count(inst)) {
                    //? in replace block list, jump
                    continue;
                }
                auto& info = mCodeGenCtx.instInfo.getInstInfo(inst);

                for (uint32_t idx = 0; idx < info.operand_num(); idx++) {
                    auto op = inst->operand(idx);
                    if (not op.isReg()) {
                        continue;
                    }
                    // replace map: old operand* -> new operand*
                    auto iter = mReplaceMap.find(op);
                    if (iter != mReplaceMap.end()) {
                        inst->set_operand(idx, iter->second);
                    }
                }
            }
        }
        if(debugISel) {
            func->print(std::cout << "after isel:\n", mCodeGenCtx);
        }
        if (modified) {
            if (debugISel) std::cout << "run_isel modified, continue!\n" << std::endl;
            continue;
        }
        // not modified, check illegal inst
        //! 检查和处理非法指令
        //! 如果存在非法指令，根据情况决定是继续尝试合法化还是报告错误。
        if (debugISel)
            std::cout << "run_isel success!" << std::endl;
        return;
    }  // while end
}

uint32_t select_copy_opcode(MIROperand dst, MIROperand src) {
    if (dst.isReg() && isISAReg(dst.reg())) {
        // dst is a isa reg
        if (src.isImm()) {
            return InstLoadImmToReg;
        }
        return InstCopyToReg;
    }
    if (src.isImm()) {
        return InstLoadImmToReg;
    }
    if (src.isReg() && isISAReg(src.reg())) {
        return InstCopyFromReg;
    }
    assert(isOperandVRegORISAReg(src) and isOperandVRegORISAReg(dst));
    return InstCopy;
}

void postLegalizeFunc(MIRFunction& func, CodeGenContext& ctx) {
  /* legalize stack operands */
  for (auto& block : func.blocks()) {
    auto& insts = block->insts();
    for (auto it = insts.begin(); it != insts.end();) {
      auto next = std::next(it);
      auto& inst = *it;
      auto& info = ctx.instInfo.getInstInfo(inst);
      for (uint32_t idx = 0; idx < info.operand_num(); idx++) {
        auto op = inst->operand(idx);
        auto lctx =
          InstLegalizeContext{inst, insts, it, ctx, std::nullopt, func};
        if (isOperandStackObject(op)) {
          if (func.stackObjs().find(op) == func.stackObjs().end()) {
            std::cerr << "stack object not found in function " << func.name()
                      << std::endl;
            assert(false);
          }
          ctx.iselInfo->legalizeInstWithStackOperand(lctx, op,
                                                     func.stackObjs().at(op));
        //   ctx.iselInfo->
        }
      }
      it = next;
    }
  }

  /* iselInfo postLegaliseInst */
  for (auto& block : func.blocks()) {
    auto& insts = block->insts();
    for (auto iter = insts.begin(); iter != insts.end(); iter++) {
      auto& inst = *iter;
      if (inst->opcode() < ISASpecificBegin) {
        ctx.iselInfo->postLegalizeInst(
          InstLegalizeContext{inst, insts, iter, ctx, std::nullopt, func});
      }
    }
  }

  ctx.target.postLegalizeFunc(func, ctx);
}

}  // namespace mir
