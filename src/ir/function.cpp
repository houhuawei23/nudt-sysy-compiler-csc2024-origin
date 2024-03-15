#include "include/function.hpp"
#include "include/utils.hpp"

namespace ir {
BasicBlock* Function::add_block() {
    auto nb = new BasicBlock("", this);
    _blocks.emplace_back(nb);
    return nb;
}
void Function::print(std::ostream& os) const {
    auto return_type = ret_type();
    // auto param_types = param_type();
    os << "define " << *return_type << " @" << name() << "(";
    // print fparams
    //! to do
    // for (auto &p : params()) {
    //     //
    // }
    os << ") {\n";
    // print bbloks
    for (auto& bb : _blocks) {
        os << *bb << std::endl;
    }
    os << "}";
}
}  // namespace ir