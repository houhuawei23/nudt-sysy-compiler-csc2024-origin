#include "f.hpp"
#include "visitor.hpp"

namespace sysy {
std::any
SysYIRGenerator::visitCompUnit(SysYParser::CompUnitContext* ctx)
{
    std::cout << "visitCompUnit" << std::endl;
    std::cout << tmp::f(5) << std::endl;
    // std:: cout << ctx->

    visitChildren(ctx);
    
    return nullptr;
}
} // namespace sysy
