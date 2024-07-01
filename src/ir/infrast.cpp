#include "ir/infrast.hpp"
#include "ir/function.hpp"
#include "ir/utils_ir.hpp"
#include "ir/instructions.hpp"

namespace ir {

void Argument::print(std::ostream& os) const{
    os << *type() << " " << name();
}

void BasicBlock::print(std::ostream& os) const{
    // print all instructions

    os << name() << ":";
    /* comment begin */
    if (not _comment.empty()) {
        os << " ; " << _comment << std::endl;
    } else {
        os << std::endl;
    }
    if (not _pre_blocks.empty()) {
        os << "    ; " << "pres: ";
        for (auto it = pre_blocks().begin(); it != pre_blocks().end(); it++) {
            os << (*it)->name();
            if (std::next(it) != pre_blocks().end()) {
                os << ", ";
            }
        }
        os << std::endl;
    }
    if (not _next_blocks.empty()) {
        os << "    ; " << "nexts: ";
        for (auto it = next_blocks().begin(); it != next_blocks().end(); it++) {
            os << (*it)->name();
            if (std::next(it) != next_blocks().end()) {
                os << ", ";
            }
        }
        os << std::endl;
    }
    /* comment end */

    for (auto& inst : _insts) {
        os << "    " << *inst << std::endl;
    }
}

void BasicBlock::emplace_back_inst(Instruction* i) {
    if (not _is_terminal)
        _insts.emplace_back(i);
    _is_terminal = i->is_terminator();
    if (auto phiInst = dyn_cast<PhiInst>(i))
        assert(false and "a phi can not be inserted at the back of a bb");
}
void BasicBlock::emplace_inst(inst_iterator pos, Instruction* i) {
    // Warning: didn't check _is_terminal
    _insts.emplace(pos, i);
    _is_terminal = i->is_terminator();
    if (auto phiInst = dyn_cast<PhiInst>(i)) {
        // assume that Phi insts are all at the front of a bb
        int index = std::distance(_insts.begin(), pos);
        _phi_insts.emplace(std::next(_phi_insts.begin(), index), phiInst);
    }
}

void Instruction::setvarname() {
    auto cur_func = _parent->parent();
    _name = "%" + std::to_string(cur_func->getvarcnt());
}

void BasicBlock::delete_inst(Instruction* inst) {
    // if inst1 use 2, 2->_uses have use user inst
    // in 2, del use of 1
    // if 3 use inst, 3.operands have use(3, 1)
    // first replace use(3, 1)
    // if you want to delete a inst, all use of it must be deleted in advance
    assert(inst->uses().size() == 0);
    for (auto op_use : inst->operands()) {
        auto op = op_use->value();
        op->uses().remove(op_use);
    }
    _insts.remove(inst);
    if (auto phiInst = dyn_cast<PhiInst>(inst))
        _phi_insts.remove(phiInst);

    // delete inst;
}

void BasicBlock::force_delete_inst(Instruction* inst) {
    // assert(inst->uses().size()==0);
    for (auto op_use : inst->operands()) {
        auto op = op_use->value();
        op->uses().remove(op_use);
    }
    _insts.remove(inst);
    if (auto phiInst = dyn_cast<PhiInst>(inst))
        _phi_insts.remove(phiInst);
}

void BasicBlock::emplace_first_inst(Instruction* inst) {
    // Warning: didn't check _is_terminal
    auto pos = _insts.begin();
    _insts.emplace(pos, inst);
    _is_terminal = inst->is_terminator();
    if (auto phiInst = dyn_cast<PhiInst>(inst))
        _phi_insts.emplace_front(phiInst);
}

void BasicBlock::replaceinst(Instruction* old_inst, Value* new_) {
    inst_iterator pos = find(_insts.begin(), _insts.end(), old_inst);
    if (pos != _insts.end()) {
        if (auto inst = dyn_cast<Instruction>(new_)) {
            emplace_inst(pos, inst);
            old_inst->replace_all_use_with(inst);
            delete_inst(old_inst);
        } else if (auto constant = dyn_cast<Constant>(new_)) {
            old_inst->replace_all_use_with(constant);
            delete_inst(old_inst);
        }
    }
}

Instruction* Instruction::copy_inst(std::function<Value*(Value*)> getValue) {
    if (auto allocainst = dyn_cast<AllocaInst>(this)) {
        if (allocainst->is_scalar())
            return new AllocaInst(
                allocainst->base_type());  // TODO 复制数组的alloca
        else {
            auto basetype = dyn_cast<ArrayType>(allocainst->base_type());
            std::vector<int> dims = basetype->dims();
            return new AllocaInst(basetype->base_type(), dims);
        }
    } else if (auto storeinst = dyn_cast<StoreInst>(this)) {
        auto value = getValue(storeinst->operand(0));
        auto addr = getValue(storeinst->operand(1));
        return new StoreInst(value, addr);
    } else if (auto loadinst = dyn_cast<LoadInst>(this)) {
        auto ptr = getValue(loadinst->ptr());
        return new LoadInst(ptr, loadinst->type(), nullptr);
    } else if (auto returninst = dyn_cast<ReturnInst>(this)) {
        return new ReturnInst(getValue(returninst->return_value()));
    } else if (auto unaryinst = dyn_cast<UnaryInst>(this)) {
        auto value = getValue(unaryinst->get_value());
        return new UnaryInst(unaryinst->scid(), unaryinst->type(), value);
    } else if (auto binaryinst = dyn_cast<BinaryInst>(this)) {
        auto lhs = getValue(binaryinst->get_lvalue());
        auto rhs = getValue(binaryinst->get_rvalue());
        return new BinaryInst(binaryinst->scid(), binaryinst->type(), lhs, rhs,
                              nullptr);
    } else if (auto callinst = dyn_cast<CallInst>(this)) {
        auto callee = callinst->callee();
        std::vector<Value*> args;
        for (auto arg : callinst->rargs()) {
            args.push_back(getValue(arg->value()));
        }
        return new CallInst(callee, args);
    } else if (auto branchinst = dyn_cast<BranchInst>(this)) {
        if (branchinst->is_cond()) {
            auto cond = getValue(branchinst->cond());
            auto true_bb = dyn_cast<BasicBlock>(getValue(branchinst->iftrue()));
            auto false_bb =
                dyn_cast<BasicBlock>(getValue(branchinst->iffalse()));
            return new BranchInst(cond, true_bb, false_bb);
        } else {
            auto dest_bb = dyn_cast<BasicBlock>(getValue(branchinst->dest()));
            return new BranchInst(dest_bb);
        }
    } else if (auto icmpinst = dyn_cast<ICmpInst>(this)) {
        auto lhs = getValue(icmpinst->lhs());
        auto rhs = getValue(icmpinst->rhs());
        return new ICmpInst(icmpinst->scid(), lhs, rhs, nullptr);
    } else if (auto fcmpinst = dyn_cast<FCmpInst>(this)) {
        auto lhs = getValue(fcmpinst->lhs());
        auto rhs = getValue(fcmpinst->rhs());
        return new FCmpInst(fcmpinst->scid(), lhs, rhs, nullptr);
    } else if (auto bitcastinst = dyn_cast<BitCastInst>(this)) {
        auto value = getValue(bitcastinst->value());
        return new BitCastInst(bitcastinst->type(), value, nullptr);
    } else if (auto memsetinst = dyn_cast<MemsetInst>(this)) {
        auto value = getValue(memsetinst->value());
        return new MemsetInst(memsetinst->type(), value, nullptr);
    } else if (auto getelemenptrinst = dyn_cast<GetElementPtrInst>(this)) {
        auto value = getValue(getelemenptrinst->get_value());
        auto idx = getValue(getelemenptrinst->get_index());
        if (getelemenptrinst->getid() == 0) {
            auto basetype = getelemenptrinst->base_type();
            return new GetElementPtrInst(basetype, value, nullptr, idx);
        } else if (getelemenptrinst->getid() == 1) {
            auto basetype = dyn_cast<ArrayType>(getelemenptrinst->base_type());
            std::vector<int> dims = basetype->dims();
            auto curdims = getelemenptrinst->cur_dims();
            return new GetElementPtrInst(basetype->base_type(), value, nullptr,
                                         idx, dims, curdims);
        } else {
            auto basetype = getelemenptrinst->base_type();
            auto curdims = getelemenptrinst->cur_dims();
            return new GetElementPtrInst(basetype, value, nullptr, idx,
                                         curdims);
        }
    } else if (auto phiinst = dyn_cast<PhiInst>(this)) {
        return new PhiInst(nullptr, phiinst->type());
    } else {
        return nullptr;
    }
}
}  // namespace ir