#include "ir/ir.hpp"

#include "mir/mir.hpp"
#include "mir/target.hpp"
#include "mir/instinfo.hpp"
#include "mir/iselinfo.hpp"

#include "target/generic/InstInfoDecl.hpp"

#include <optional>

namespace mir {

std::unordered_map<MIROperand*, uint32_t> collect_def_count(
    MIRFunction* func,
    CodeGenContext& codegen_ctx) {
    std::cerr << "collect_def_count not implemented yet!" << std::endl;
    return std::unordered_map<MIROperand*, uint32_t>();
}

void ISelContext::remove_inst(MIRInst* inst) {
    assert(inst != nullptr);
    _remove_work_list.insert(inst);
}
void ISelContext::replace_operand(MIROperand* src, MIROperand* dst) {
    assert(src->is_reg());
    if (src != dst) {
        _replace_map.emplace(src, dst);
    }
}
MIROperand* ISelContext::get_inst_def(MIRInst* inst) {
    assert(inst != nullptr);
    auto& instinfo = _codegen_ctx.instInfo.get_instinfo(inst);
    for (uint32_t idx = 0; idx < instinfo.operand_num(); idx++) {
        if (instinfo.operand_flag(idx) & OperandFlagDef) {
            return inst->operand(idx);
        }
    }
    // assert(false && "no def operand found");
    std::cerr << "no def operand found" << std::endl;
    return nullptr;
}

MIRInst* ISelContext::lookup_def(MIROperand* op) {
    assert(op != nullptr);
    auto iter = _inst_map.find(op);
    if (iter != _inst_map.end()) {
        return iter->second;
    }

    assert(false && "def not found");
}

void ISelContext::run_isel(MIRFunction* func) {
    // TODO: implement
    auto& isel_info = _codegen_ctx.iselInfo;

    //! fix point algorithm: 循环执行指令选择和替换，直到不再变化。
    while (true) {
        std::cout << "[run_isel] fix point while loop start" << std::endl;
        bool modified = false;
        bool has_illegal = false;
        _remove_work_list.clear();
        _replace_block_list.clear();
        _replace_map.clear();

        _constant_map.clear();
        _use_cnt.clear();
        _inst_map.clear(); //! here!
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
        for (auto& block : func->blocks()) {
            ir::BasicBlock* ir_block = block->ir_block();
            std::cout << "for all ir_block: ";
            std::cout << ir_block->name() << std::endl;
            // check ssa form, get inst map
            for (auto& inst : block->insts()) {
                auto& instinfo = _codegen_ctx.instInfo.get_instinfo(inst);
                for (uint32_t idx = 0; idx < instinfo.operand_num(); idx++) {
                    if (instinfo.operand_flag(idx) & OperandFlagDef) {
                        auto def = inst->operand(idx);
                        if (def->is_reg() && isVirtualReg(def->reg())) {
                            _inst_map.emplace(def, inst);
                        }
                    }
                }
            }
            _curr_block = block.get();
            auto& insts = block->insts();

            if (insts.empty())
                continue;

            auto it = std::prev(insts.end());
            /* in this block */
            while (true) {  // for all insts in block
                _insert_point = it;
                auto& inst = *it;
                std::optional<std::list<MIRInst*>::iterator> prev;

                if (it != insts.begin()) {
                    prev = std::prev(it);
                }
                // if inst not in remove list
                if (not _remove_work_list.count(inst)) {
                    std::cout << "  [match&select] inst: " << inst->opcode()
                              << std::endl;
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
                [&](auto inst) { return _remove_work_list.count(inst); });

            // replace defs
            for (auto& inst : block->insts()) {
                if (_replace_block_list.count(inst)) {
                    //? in replace block list, jump
                    continue;
                }
                auto& info = _codegen_ctx.instInfo.get_instinfo(inst);

                for (uint32_t idx = 0; idx < info.operand_num(); idx++) {
                    auto op = inst->operand(idx);
                    if (not op->is_reg()) {
                        continue;
                    }
                    // replace map: old operand* -> new operand*
                    auto iter = _replace_map.find(op);
                    if (iter != _replace_map.end()) {
                        inst->set_operand(idx, iter->second);
                    }
                }
            }
        }

        if (modified) {
            std::cout << "run_isel modified, continue!\n" << std::endl;
            continue;
        }
        // not modified, check illegal inst
        //! 检查和处理非法指令
        //! 如果存在非法指令，根据情况决定是继续尝试合法化还是报告错误。

        std::cout << "run_isel success!" << std::endl;
        return;
    }  // while end
}

uint32_t select_copy_opcode(MIROperand* dst, MIROperand* src) {
    if (dst->is_reg() && isISAReg(dst->reg())) {
        // dst is a isa reg
        if (src->is_imm()) {
            return InstLoadImmToReg;
        }
        return InstCopyToReg;
    }
    if (src->is_imm()) {
        return InstLoadImmToReg;
    }
    if (src->is_reg() && isISAReg(src->reg())) {
        return InstCopyFromReg;
    }
    assert(isOperandVRegORISAReg(src) and isOperandVRegORISAReg(dst));
    return InstCopy;
}

}  // namespace mir
