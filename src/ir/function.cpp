#include "include/function.hpp"

namespace ir {
BasicBlock *Function::add_bblock(const std::string &name) {
    auto nb = new BasicBlock(this, name);
    _blocks.emplace_back(nb);
    return nb;
}

} // namespace ir