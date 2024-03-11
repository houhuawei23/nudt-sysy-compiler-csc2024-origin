#include "include/function.hpp"
#include "include/utils.hpp"

namespace ir {
BasicBlock *Function::add_bblock(const std::string &name) {
    auto nb = new BasicBlock(this, name);
    _blocks.emplace_back(nb);
    return nb;
}
void Function::print(std::ostream &os) const {
    auto ret_type = get_ret_type();
    // auto param_types = get_param_type();
    os << "define " << *ret_type << " @" << get_name() << "(";
    // print fparams
    //! to do
    // for (auto &p : get_params()) {
    //     //
    // }
    os << ") {\n";
    // print bbloks
    for (auto &bb : _blocks) {
        os << *bb << std::endl;
    }
    os << "}";
}
} // namespace ir