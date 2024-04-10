#include "visitor/visitor.hpp"

#include <stdbool.h>
namespace sysy {
std::any SysYIRGenerator::visitCompUnit(SysYParser::CompUnitContext* ctx) {
    ir::SymbolTableBeta::ModuleScope scope(_tables);
    // TODO: add runtime lib functions
    auto type_i32 = ir::Type::i32_type();
    auto type_f32 = ir::Type::float_type();
    auto type_void = ir::Type::void_type();
    auto type_i32p = ir::Type::pointer_type(type_i32);
    auto type_f32p = ir::Type::pointer_type(type_f32);


    //! 外部函数
    module()->add_func(ir::Type::func_type(type_i32, {}), "getint");
    module()->add_func(ir::Type::func_type(type_i32, {}), "getch");
    module()->add_func(ir::Type::func_type(type_f32, {}), "getfloat");

    module()->add_func(ir::Type::func_type(type_i32, {type_i32p}), "getarray");
    module()->add_func(ir::Type::func_type(type_i32, {type_f32p}), "getfarray");


    module()->add_func(ir::Type::func_type(type_void, {type_i32}), "putint");
    module()->add_func(ir::Type::func_type(type_void, {type_i32}), "putch");
    module()->add_func(ir::Type::func_type(type_void, {type_f32}), "putfloat");

    module()->add_func(ir::Type::func_type(type_void, {type_i32, type_i32p}), "putarray");
    module()->add_func(ir::Type::func_type(type_void, {type_i32, type_f32p}), "putfarray");

    module()->add_func(ir::Type::func_type(type_void, {}), "putf");

    module()->add_func(ir::Type::func_type(type_void, {}), "starttime");
    module()->add_func(ir::Type::func_type(type_void, {}), "stoptime");

    visitChildren(ctx);
    return nullptr;
}
}  // namespace sysy
