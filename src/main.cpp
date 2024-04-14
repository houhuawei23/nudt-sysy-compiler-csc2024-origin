// clang-format off
#include <iostream>
#include "SysYLexer.h"
#include "visitor/visitor.hpp"

#include "pass/pass.hpp"
#include "pass/analysis/dom.hpp"
#include "pass/optimize/mem2reg.hpp"

#include "mir/mir.hpp"
#include "mir/target.hpp"
#include "mir/lowering.hpp"

#include "target/riscv.hpp"
#include "target/riscvtarget.hpp"
// clang-format on
#include "pass/optimize/DCE.hpp"
using namespace std;

int main(int argc, char** argv) {
    if (argc != 2) {
        cerr << "Usage: " << argv[0] << "inputfile\n";
        return EXIT_FAILURE;
    }
    ifstream fin(argv[1]);
    if (not fin) {
        cerr << "Failed to open file " << argv[1];
        return EXIT_FAILURE;
    }
    antlr4::ANTLRInputStream input(fin);
    SysYLexer lexer(&input);
    antlr4::CommonTokenStream tokens(&lexer);
    SysYParser parser(&tokens);

    SysYParser::CompUnitContext* ast_root = parser.compUnit();

    // IR Generation
    ir::Module* base_module = new ir::Module();
    sysy::SysYIRGenerator gen(base_module, ast_root);  // forget to pass module
    gen.build_ir();

    auto module_ir = gen.module();
    bool genir = true;
    if (genir) {
        module_ir->print(std::cout);
    }

    // pass::FunctionPassManager fpm;
    // //mem2reg
    // fpm.add_pass(new pass::preProcDom());
    // fpm.add_pass(new pass::idomGen());
    // fpm.add_pass(new pass::domFrontierGen());
    // // fpm.add_pass(new pass::domInfoCheck());
    // fpm.add_pass(new pass::Mem2Reg());
    // for(auto f : module_ir->funcs()){
    //     fpm.run(f);
    // }

    // if (genir) {
    //     module_ir->print(std::cout);
    // }

    auto target = mir::RISCVTarget();
    auto mir_module = mir::create_mir_module(*module_ir, target);
    pass::FunctionPassManager fpm;
    // //mem2reg
    // fpm.add_pass(new pass::preProcDom());
    // fpm.add_pass(new pass::idomGen());
    // // fpm.add_pass(new pass::domFrontierGen());
    // fpm.add_pass(new pass::domInfoCheck());
    fpm.add_pass(new pass::Mem2Reg());
    fpm.add_pass(new pass::DCE());
    for(auto f : module_ir->funcs()){
        fpm.run(f);
    }

    if (genir) {
        module_ir->print(std::cout);
    }
    
    // MIR Generation
    // mir::MIRModule* mir_base_module = new mir::MIRModule(module_ir);
    // bool gen_mir = true;
    // if (gen_mir) {
    //     mir_base_module->print(std::cout);
    // }

    return EXIT_SUCCESS;
}