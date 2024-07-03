#include "ir/function.hpp"
#include "ir/utils_ir.hpp"
#include "ir/type.hpp"
#include "ir/instructions.hpp"

namespace ir {
BasicBlock* Function::new_block() {
    auto nb = new BasicBlock("", this);
    _blocks.emplace_back(nb);
    return nb;
}

void Function::delete_block(BasicBlock* bb) {
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
    _blocks.remove(bb);
    // delete bb;
}

void Function::force_delete_block(BasicBlock* bb) {
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
    _blocks.remove(bb);
}

void Function::print(std::ostream& os) const {
    auto return_type = ret_type();
    if (blocks().size()) {
        os << "define " << *return_type << " @" << name() << "(";
        if (_args.size() > 0) {
            auto last_iter = _args.end() - 1;
            for (auto iter = _args.begin(); iter != last_iter; ++iter) {
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
        if (auto func_type = dyn_cast<FunctionType>(t)) {
            auto args_types = func_type->arg_types();
            if (args_types.size() > 0) {
                auto last_iter = args_types.end() - 1;
                for (auto iter = args_types.begin(); iter != last_iter;
                     ++iter) {
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
        for (auto& bb : _blocks) {
            os << *bb << std::endl;
        }
        os << "}";
    } else {
        os << "\n";
    }
}

void Function::rename() {
    if(_blocks.empty()) 
        return;
    setvarcnt(0);
    for (auto arg : _args) {
        std::string argname = "%" + std::to_string(getvarcnt());
        arg->set_name(argname);
    }
size_t blockIdx = 0;
    for (auto bb : _blocks) {
        bb->set_idx(blockIdx);
        blockIdx++;
        for (auto inst : bb->insts()) {
            if (inst->is_noname())
                continue;
            auto callpt = dyn_cast<CallInst>(inst);
            if (callpt and callpt->is_void())
                continue;
            inst->setvarname();
        }
    }
}

// func_copy
Function* Function::copy_func() {
    std::unordered_map<Value*, Value*> ValueCopy;
    // copy global
    for (auto gvalue : _parent->gvalues()) {
        if (dyn_cast<Constant>(gvalue) && !gvalue->type()->is_pointer()) {
            ValueCopy[gvalue] = gvalue;  //??
        } else {
            ValueCopy[gvalue] = gvalue;
        }
    }
    // copy func
    auto copyfunc = new Function(type(), name() + "_copy", _parent);
    // copy args
    for (auto arg : _args) {
        Value* copyarg = copyfunc->new_arg(arg->type(), "");
        ValueCopy[arg] = copyarg;
    }
    // copy block
    for (auto bb : _blocks) {
        BasicBlock* copybb = copyfunc->new_block();
        if (copyfunc->entry() == nullptr) {
            copyfunc->setEntry(copybb);
        } else if (copyfunc->exit() == nullptr) {
            copyfunc->setExit(copybb);
        }
        ValueCopy[bb] = copybb;
    }

    // copy bb's pred and succ
    for (auto BB : _blocks) {
        auto copyBB = dyn_cast<BasicBlock>(ValueCopy[BB]);
        for (auto pred : BB->pre_blocks()) {
            copyBB->pre_blocks().emplace_back(
                dyn_cast<BasicBlock>(ValueCopy[pred]));
        }
        for (auto succ : BB->next_blocks()) {
            copyBB->next_blocks().emplace_back(
                dyn_cast<BasicBlock>(ValueCopy[succ]));
        }
    }

    auto getValue = [&](Value* val) -> Value* {
        if (auto c = dyn_cast<Constant>(val))
            return c;
        return ValueCopy[val];
    };

    // copy inst in bb
    std::vector<PhiInst*> phis;
    std::set<BasicBlock*> vis;
    BasicBlock::BasicBlockDfs(_entry, [&](BasicBlock* bb) -> bool {
        if (vis.count(bb))
            return true;
        vis.insert(bb);
        auto bbCpy = dyn_cast<BasicBlock>(ValueCopy[bb]);
        for (auto inst : bb->insts()) {
            auto copyinst = inst->copy_inst(getValue);
            copyinst->set_parent(bbCpy);
            ValueCopy[inst] = copyinst;
            bbCpy->insts().emplace_back(copyinst);
            if (auto phi = dyn_cast<PhiInst>(inst))
                phis.emplace_back(phi);
        }
        return false;
    });
    for (auto phi : phis) {
        auto copyphi = dyn_cast<PhiInst>(ValueCopy[phi]);
        for (size_t i = 0; i < phi->getsize(); i++) {
            auto phivalue = getValue(phi->getval(i));
            auto phibb = dyn_cast<BasicBlock>(getValue(phi->getbb(i)));
            copyphi->addIncoming(phivalue, phibb);
        }
    }
    return copyfunc;
}

}  // namespace ir