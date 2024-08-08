#include "pass/optimize/optimize.hpp"
#include "pass/optimize/GVN.hpp"
#include "ir/value.hpp"
#include <set>
#include <cassert>
#include <map>
#include <vector>
#include <queue>
#include <algorithm>

namespace pass {
void GVN::dfs(ir::BasicBlock* bb) {
    assert(bb != nullptr && "nullptr in GVN");
    visited.insert(bb);
    for (auto succ : bb->next_blocks()) {
        if (visited.find(succ) == visited.end()) {
            dfs(succ);
        }
    }
    RPOblocks.push_back(bb);
}

void GVN::RPO(ir::Function* F) {
    RPOblocks.clear();
    visited.clear();
    auto root = F->entry();
    // assert(root != nullptr && "Function without entry block");
    dfs(root);
    reverse(RPOblocks.begin(), RPOblocks.end());
}

ir::Value* GVN::getValueNumber(ir::Instruction* inst) {
    if (auto binary = inst->dynCast<ir::BinaryInst>())
        return getValueNumber(binary);
    else if (auto unary = inst->dynCast<ir::UnaryInst>())
        return getValueNumber(unary);
    else if (auto getelementptr = inst->dynCast<ir::GetElementPtrInst>())
        return getValueNumber(getelementptr);
    else if (auto load = inst->dynCast<ir::LoadInst>())
        return getValueNumber(load);
    else if (auto call = dynamic_cast<ir::CallInst*>(inst)) {
        auto callee = call->callee();
        if (sectx->isPureFunc(callee)) {
            return getValueNumber(call);
        }
        return nullptr;
    } else
        return nullptr;
}

ir::Value* GVN::getValueNumber(ir::BinaryInst* inst) {
    auto lhs = checkHashtable(inst->lValue());
    auto rhs = checkHashtable(inst->rValue());
    for (auto [Key, Value] : _Hashtable) {
        if (auto binary = dynamic_cast<ir::BinaryInst*>(Key)) {
            auto binlhs = checkHashtable(binary->lValue());
            auto binrhs = checkHashtable(binary->rValue());
            if (binary->valueId() == inst->valueId() && ((lhs == binlhs && rhs == binrhs) || (binary->isCommutative() && lhs == binrhs && rhs == binlhs))) {
                return Value;
            }
        }
    }
    return inst;
}

ir::Value* GVN::getValueNumber(ir::UnaryInst* inst) {
    auto val = checkHashtable(inst->value());
    for (auto [Key, Value] : _Hashtable) {
        if (auto unary = dynamic_cast<ir::UnaryInst*>(Key)) {
            auto unval = checkHashtable(unary->value());
            if (unary->valueId() == inst->valueId() && unval == val) return Value;
        }
    }
    return inst;
}

ir::Value* GVN::getValueNumber(ir::GetElementPtrInst* inst) {
    auto arval = checkHashtable(inst->value());
    auto aridx = checkHashtable(inst->index());
    for (auto [Key, Value] : _Hashtable) {
        if (auto getelementptr = dynamic_cast<ir::GetElementPtrInst*>(Key)) {
            auto getval = checkHashtable(getelementptr->value());
            auto getidx = checkHashtable(getelementptr->index());

            if (arval == getval && aridx == getidx) {
                return Value;
            }
        }
    }
    return inst;
}

ir::Value* GVN::getValueNumber(ir::LoadInst* inst) {
    auto arval = checkHashtable(inst->ptr());
    bool artype = inst->type()->isPointer();
    for (auto [Key, Value] : _Hashtable) {
        if (auto load = Key->dynCast<ir::LoadInst>()) {
            auto getval = checkHashtable(load->ptr());
            bool artype = load->type()->isPointer();
            if (arval == getval && artype && artype) {
                return Value;
            }
        }
    }
    return inst;
}

ir::Value* GVN::getValueNumber(ir::CallInst* inst) {
    for (auto [Key, Value] : _Hashtable) {
        if (auto call = Key->dynCast<ir::CallInst>()) {
            bool flag = true;
            if (call->callee() == inst->callee()) {
                for (auto arg : inst->rargs()) {
                    auto instarg = checkHashtable(arg->value());
                    auto callarg = checkHashtable(call->operand(arg->index()));
                    if (instarg != callarg) {
                        flag = false;
                        break;
                    }
                }
                if (flag) return Value;
            }
        }
    }
    return static_cast<ir::Value*>(inst);
}

ir::Value* GVN::checkHashtable(ir::Value* v) {
    if (auto vnum = _Hashtable.find(v); vnum != _Hashtable.end()) {
        return vnum->second;
    }
    if (auto inst = v->dynCast<ir::Instruction>()) {
        if (auto value = getValueNumber(inst)) {
            _Hashtable[v] = value;
            return value;
        }
    }
    _Hashtable[v] = v;
    return v;
}

void GVN::visitinst(ir::Instruction* inst) {
    auto bb = inst->block();
    for (auto use : inst->uses()) {
        if (auto br = dyn_cast<ir::BranchInst>(use->user())) {
            return;
        }
    }

    auto value = checkHashtable(inst);
    if (inst != value) {
        if (auto instvalue = dyn_cast<ir::Instruction>(value)) {
            auto vbb = instvalue->block();
            if (domctx->dominate(vbb, bb))  // vbb->dominate(bb)
            {
                inst->replaceAllUseWith(instvalue);
                NeedRemove.insert(inst);
            }
        }
    }
    return;
}

void GVN::run(ir::Function* F, TopAnalysisInfoManager* tp) {
    if (F->blocks().empty()) return;

    RPOblocks.clear();
    visited.clear();
    NeedRemove.clear();
    _Hashtable.clear();

    domctx = tp->getDomTree(F);
    sectx = tp->getSideEffectInfo();
    RPO(F);
    visited.clear();
    for (auto bb : RPOblocks) {
        for (auto inst : bb->insts()) {
            visitinst(inst);
        }
    }
    for (auto inst : NeedRemove) {
        auto BB = inst->block();
        BB->delete_inst(inst);
    }
}

}  // namespace pass