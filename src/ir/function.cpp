#include "include/function.hpp"
#include "include/utils.hpp"

namespace ir {
BasicBlock* Function::new_block() {
    auto nb = new BasicBlock("", this);
    _blocks.emplace_back(nb);
    return nb;
}

void Function::print(std::ostream& os) const {
    auto return_type = ret_type();
    // auto param_types = param_type();
    os << "define " << *return_type << " @" << name() << "(";
    
    // print fparams
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
    
    os << ") {\n";
    
    // print bbloks
    for (auto& bb : _blocks) {
        if (!bb->empty()) {
            os << *bb << std::endl;
        }
    }
    os << "}";
}
}  // namespace ir