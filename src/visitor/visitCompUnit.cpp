#include "visitor/visitor.hpp"

#include <stdbool.h>
namespace sysy {
std::any SysYIRGenerator::visitCompUnit(SysYParser::CompUnitContext* ctx) {
    ir::SymbolTableBeta::ModuleScope scope(_tables);
    // TO DO: add runtime lib functions
    auto type_i32 = ir::Type::i32_type();
    auto type_f64 = ir::Type::double_type();
    auto type_void = ir::Type::void_type();
    // ir::Type* func_type = ir::Type::function_type(, {});
    // auto getIntFunc = 

    //! 外部函数
    module()->add_function(ir::Type::function_type(type_i32, {}), "getint");
    module()->add_function(ir::Type::function_type(type_i32, {}), "getch");
    module()->add_function(ir::Type::function_type(type_f64, {}), "getfloat");

    module()->add_function(ir::Type::function_type(type_i32, {}), "getarray"); //!
    module()->add_function(ir::Type::function_type(type_i32, {}), "getfarray"); //!
  
    ir::type_ptr_vector v1;
    v1.push_back(type_i32);
    ir::type_ptr_vector v2;
    v2.push_back(type_f64);

    module()->add_function(ir::Type::function_type(type_void, v1), "putint");
    module()->add_function(ir::Type::function_type(type_void, v1), "putch");
    module()->add_function(ir::Type::function_type(type_void, v2), "putfloat");


    module()->add_function(ir::Type::function_type(type_void, v2), "putarray");
    module()->add_function(ir::Type::function_type(type_void, v2), "putfarray");

    module()->add_function(ir::Type::function_type(type_void, v2), "putf");
    
    module()->add_function(ir::Type::function_type(type_void, {}), "starttime");
    module()->add_function(ir::Type::function_type(type_void, {}), "stoptime");
    
    
    
    visitChildren(ctx);
    return nullptr;
}
}  // namespace sysy
