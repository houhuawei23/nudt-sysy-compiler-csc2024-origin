#include "ir/function.hpp"
#include "ir/utils_ir.hpp"
#include "ir/type.hpp"

namespace ir {
BasicBlock* Function::new_block() {
    auto nb = new BasicBlock("", this);
    _blocks.emplace_back(nb);
    return nb;
}

void Function::delete_block(BasicBlock* bb){
    for(auto bbpre:bb->pre_blocks()){
        bbpre->next_blocks().remove(bb);
    }
    for(auto bbnext:bb->next_blocks()){
        bbnext->pre_blocks().remove(bb);
    }
    for(auto bbinstIter=bb->insts().begin();bbinstIter!=bb->insts().end();){
        auto delinst=*bbinstIter;
        bbinstIter++;
        bb->delete_inst(delinst);
    }
    _blocks.remove(bb);
    delete bb;
}

void Function::print(std::ostream& os) {
    auto return_type = ret_type();
    setvarcnt(0);
    
    if (blocks().size()) {
        os << "define " << *return_type << " @" << name() << "(";
        if (_args.size() > 0) {
            auto last_iter = _args.end() - 1;
            for (auto iter = _args.begin(); iter != last_iter; ++iter) {
                auto arg = *iter;
                arg->setname("%" + std::to_string(getvarcnt()));
                os << *(arg->type()) << " " << arg->name();
                os << ", ";
            }
            auto arg = *last_iter;
            arg->setname("%" + std::to_string(getvarcnt()));
            os << *(arg->type()) << " " << arg->name();
        }
    } else {
        os << "declare " << *return_type << " @" << name() << "(";
        auto t = type();
        if (auto func_type = dyn_cast<FunctionType>(t)) {
            auto args_types = func_type->arg_types();
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
        for (auto& bb : _blocks) {
            os << *bb << std::endl;
        }
        os << "}";
    } else {
        os << "\n";
    }
}
}  // namespace ir