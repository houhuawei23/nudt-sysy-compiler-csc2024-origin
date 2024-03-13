#include "visitor.hpp"


namespace sysy {
std::any SysYIRGenerator::visitCompUnit(SysYParser::CompUnitContext* ctx) {
    ir::SymbolTableBeta::ModuleScope scope(_tables);

    // TO DO: add runtime lib functions
    visitChildren(ctx);
    return nullptr;
}
}  // namespace sysy
