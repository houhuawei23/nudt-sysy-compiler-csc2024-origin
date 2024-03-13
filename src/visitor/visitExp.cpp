#include "visitor.hpp"

namespace sysy {
std::any SysYIRGenerator::visitNumberExp(SysYParser::NumberExpContext* ctx) {
    // number: ILITERAL | FLITERAL;

    ir::Value* res = nullptr;
    if (auto iLiteral = ctx->number()->ILITERAL()) {  // int
        std::string s = iLiteral->getText();
        // dec
        int base = 10;
        // hex
        if (s.length() > 2 && s[0] == '0' && (s[1] == 'x' || s[1] == 'X')) {
            base = 16;

        } else if (s[0] == '0') {  // oct
            base = 8;
        }

        res = ir::Constant::gen(std::stoi(s, 0, base));
    } else if (auto fctx = ctx->number()->FLITERAL()) {  // float
        std::string s = fctx->getText();
        res = ir::Constant::gen(std::stof(s)); // stod?
    }
    return res;
}

}  // namespace sysy
