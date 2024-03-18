#include "visitor.hpp"

#include <stdbool.h>
namespace sysy {
std::any SysYIRGenerator::visitCompUnit(SysYParser::CompUnitContext* ctx) {
    ir::SymbolTableBeta::ModuleScope scope(_tables);
    //test
    // auto c0 = ir::Constant::gen_i32(0);
    // // ir::Constant::gen(0);
    // auto c1 = ir::Constant::gen_i32(1);
    // auto ct = ir::Constant::gen_i1(true);
    // auto cf = ir::Constant::gen_i1(false);

    // auto cc0 = ir::Constant::gen_i32(0);
    // // ir::Constant::gen(0);
    // auto cc1 = ir::Constant::gen_i32(1);

    // auto f0 = ir::Constant::gen_f64(0);
    // auto f1 = ir::Constant::gen_f64(1);
    
    // auto cct = ir::Constant::gen(true);
    // auto ccf = ir::Constant::gen(false);
    // TO DO: add runtime lib functions
    visitChildren(ctx);
    return nullptr;
}
}  // namespace sysy
