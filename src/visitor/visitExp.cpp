#include "visitor.hpp"

namespace sysy {
std::any
SysYIRGenerator::visitNumberExp(SysYParser::NumberExpContext *ctx) {
    // number: ILITERAL | FLITERAL;
    // just for int dec
    ir::Value* res = nullptr;
    if (auto iLiteral = ctx->number()->ILITERAL()) {
        std::string text = iLiteral->getText(); 
        int base = 10;
        res = ir::Constant::gen(std::stoi(text, 0, base));
    }
    else{
        // float
    }
    return res;
}

} // namespace sysy
