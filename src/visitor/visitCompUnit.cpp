// #include "f.hpp"
#include "visitor.hpp"

// #include "infrast.hpp"
#include "module.hpp"

namespace sysy {
std::any SysYIRGenerator::visitCompUnit(SysYParser::CompUnitContext *ctx) {
    // std::cout << "visitCompUnit" << std::endl;
    // std::cout << tmp::f(5) << std::endl;
    // std::cout << ctx->getText() << std::endl;
    ir::Module module;
    visitChildren(ctx);

    return nullptr;
}
} // namespace sysy
