// clang-format off
#include <iostream>

#include "support/config.hpp"

#include "SysYLexer.h"
#include "visitor/visitor.hpp"

#include "pass/pass.hpp"
#include "pass/analysis/dom.hpp"
#include "pass/optimize/mem2reg.hpp"
#include "pass/optimize/DCE.hpp"
#include "pass/optimize/SCP.hpp"
#include "pass/optimize/SCCP.hpp"

#include "mir/mir.hpp"
#include "mir/target.hpp"
#include "mir/lowering.hpp"

#include "target/riscv.hpp"
#include "target/riscvtarget.hpp"

using namespace std;

/*
 * @brief: main
 *      ./main -f test.c -i -t mem2reg -o gen.ll -O0 -L0
 */
int main(int argc, char* argv[]) {
    sysy::Config config;
    config.parse_cmd_args(argc, argv);
    config.print_info();

    ifstream fin(config.infile);

    antlr4::ANTLRInputStream input(fin);
    SysYLexer lexer(&input);
    antlr4::CommonTokenStream tokens(&lexer);
    SysYParser parser(&tokens);

    SysYParser::CompUnitContext* ast_root = parser.compUnit();

    //! 1. IR Generation
    ir::Module* base_module = new ir::Module();
    sysy::SysYIRGenerator gen(base_module, ast_root);
    auto module_ir = gen.build_ir();

    //! 2. Optimization Passes
    pass::FunctionPassManager fpm;

    if (not config.pass_names.empty()) {
        for (auto pass_name : config.pass_names) {
            if (pass_name.compare("mem2reg") == 0) {
                // mem2reg
                fpm.add_pass(new pass::preProcDom());
                fpm.add_pass(new pass::idomGen());
                fpm.add_pass(new pass::domFrontierGen());
                fpm.add_pass(new pass::Mem2Reg());
                // fpm.add_pass(new pass::domInfoCheck());
            }

            if (pass_name.compare("dce") == 0) {
                fpm.add_pass(new pass::DCE());
            }
            if (pass_name.compare("scp") == 0) {
                fpm.add_pass(new pass::SCP());
            }
            if (pass_name.compare("sccp")==0){
                fpm.add_pass(new pass::SCCP());
            }
        }
        for (auto f : module_ir->funcs()) {
            fpm.run(f);
        }
    }

    if (config.gen_ir) {  // ir print
        if (config.outfile.empty()) {
            module_ir->print(std::cout);
        } else {
            ofstream fout;
            fout.open(config.outfile);
            module_ir->print(fout);
        }
    }

    //! 3. Code Generation
    if (config.gen_asm) {
        auto target = mir::RISCVTarget();
        auto mir_module = mir::create_mir_module(*module_ir, target);
    }

    return EXIT_SUCCESS;
}