#include <iostream>
#include "support/config.hpp"
#include "SysYLexer.h"
#include "visitor/visitor.hpp"
#include "pass/pass.hpp"
#include "pass/analysis/dom.hpp"
#include "pass/analysis/CFGAnalysis.hpp"
#include "pass/optimize/mem2reg.hpp"
#include "pass/optimize/DCE.hpp"
#include "pass/optimize/SCP.hpp"
#include "pass/optimize/mySCCP.hpp"
#include "pass/optimize/simplifyCFG.hpp"
#include "pass/analysis/loop.hpp"
#include "pass/optimize/GCM.hpp"
#include "pass/optimize/GVN.hpp"
#include "pass/analysis/pdom.hpp"
#include "pass/optimize/inline.hpp"
#include "pass/optimize/reg2mem.hpp"
#include "pass/optimize/ADCE.hpp"
#include "pass/optimize/loopsimplify.hpp"
#include "pass/analysis/irtest.hpp"
#include "pass/analysis/ControlFlowGraph.hpp"

#include "pass/optimize/InstCombine/ArithmeticReduce.hpp"

#include "mir/mir.hpp"
#include "mir/target.hpp"
#include "mir/lowering.hpp"
#include "target/riscv/riscv.hpp"
#include "target/riscv/riscvtarget.hpp"
using namespace std;
/* ./main -f test.c -i -t mem2reg -o gen.ll -O0 -L0 */
int main(int argc, char* argv[]) {
    sysy::Config config;
    config.parse_cmd_args(argc, argv);
    config.print_info();

    if (config.infile.empty()) {
        cerr << "Error: input file not specified" << endl;
        config.print_help();
        return EXIT_FAILURE;
    }
    ifstream fin(config.infile);

    antlr4::ANTLRInputStream input(fin);
    SysYLexer lexer(&input);
    antlr4::CommonTokenStream tokens(&lexer);
    SysYParser parser(&tokens);

    SysYParser::CompUnitContext* ast_root = parser.compUnit();

    //! 1. IR Generation
    auto module = ir::Module();
    ir::Module* base_module = &module;
    sysy::SysYIRGenerator gen(base_module, ast_root);
    auto module_ir = gen.build_ir();
    if(not module_ir->verify(std::cerr)) {
        std::cerr << "IR verification failed" << std::endl;
        assert(false && "IR verification failed");
    }
    //! 2. Optimization Passes
    pass::topAnalysisInfoManager* tAIM = new pass::topAnalysisInfoManager(module_ir);
    tAIM->initialize();
    pass::PassManager* pm = new pass::PassManager(module_ir,tAIM);
    pm->run(new pass::CFGAnalysisHHW());

    if (not config.pass_names.empty()) {
        for (auto pass_name : config.pass_names) {
            if (pass_name.compare("dom") == 0) {
                pm->run(new pass::domInfoPass());
                // pm->run(new pass::domInfoCheck());
            }
            else if (pass_name.compare("mem2reg") == 0){
                pm->run(new pass::Mem2Reg());
            }
            else if (pass_name.compare("pdom") == 0){
                pm->run(new pass::postDomInfoPass());
                // pm->run(new pass::postDomInfoCheck());
            }
            else if (pass_name.compare("dce") == 0) {
                pm->run(new pass::DCE());
            } else if (pass_name.compare("scp") == 0) {
                pm->run(new pass::SCP());
            } else if (pass_name.compare("sccp") == 0) {
                pm->run(new pass::mySCCP());
            } else if (pass_name.compare("simplifycfg") == 0) {
                pm->run(new pass::simplifyCFG());
            } else if (pass_name.compare("loopanalysis") == 0) {
                pm->run(new pass::loopAnalysis());
                // pm->run(new pass::loopInfoCheck());
            } else if (pass_name.compare("gcm") == 0) {
                pm->run(new pass::GCM());
            } else if (pass_name.compare("gvn") == 0) {
                pm->run(new pass::GVN());
            } else if (pass_name.compare("reg2mem") == 0){
                pm->run(new pass::Reg2Mem());
            } else if (pass_name.compare("inline") == 0){
                pm->run(new pass::Inline());
            } else if (pass_name.compare("adce") == 0){
                pm->run(new pass::ADCE());
            } else if (pass_name.compare("loopsimplify") == 0){
                pm->run(new pass::loopsimplify());
            } else if(pass_name.compare("instcombine") == 0){
                // recommend: -p mem2reg -p instcombine -p dce
                pm->run(new pass::ArithmeticReduce());
            } else if (pass_name.compare("test") == 0){
                pm->run(new pass::irCheck());
            }
            else {
                assert(false && "Invalid pass name");
            }
        }
    }
    // module_ir->print(std::cout);
    module_ir->rename();
    if (config.gen_ir) {  // ir print
        if (config.outfile.empty()) {
            module_ir->print(std::cout);
        } else {
            ofstream fout;
            fout.open(config.outfile);
            module_ir->print(fout);
        }
    }
    auto preName = [](const std::string& filePath) {
        size_t lastSlashPos = filePath.find_last_of("/\\");
        if (lastSlashPos == std::string::npos) {
            lastSlashPos = -1;  // 如果没有找到 '/', 则从字符串开头开始
        }

        // 找到最后一个 '.' 的位置
        size_t lastDotPos = filePath.find_last_of('.');
        if (lastDotPos == std::string::npos || lastDotPos < lastSlashPos) {
            lastDotPos = filePath.size();  // 如果没有找到 '.', 则到字符串末尾
        }

        // 提取 '/' 和 '.' 之间的子字符串
        return filePath.substr(lastSlashPos + 1, lastDotPos - lastSlashPos - 1);
    };

    //! 3. Code Generation
    // for (auto fun : module_ir->funcs()) {
    //     if(fun->isOnlyDeclare())continue;
    //     auto dom_ctx = tAIM->getDomTree(fun);
    //     dom_ctx->refresh();
    //     dom_ctx->BFSDomTreeInfoRefresh();
    //     for (auto bb : dom_ctx->BFSDomTreeVector()) {
    //         std::cout << bb->name() << std::endl;
    //     }
    // }
    if (config.gen_asm) {
        auto target = mir::RISCVTarget();
        // auto target = mir::GENERICTarget();
        auto mir_module = mir::create_mir_module(*module_ir, target);
        if (config.outfile.empty()) {
            target.emit_assembly(std::cout, *mir_module);
        } else {
            ofstream fout;
            fout.open(config.outfile);
            target.emit_assembly(fout, *mir_module);
        }
        {
            ofstream fout("./.debug/" + preName(config.infile) + ".s");
            target.emit_assembly(fout, *mir_module);
        }
    }

    return EXIT_SUCCESS;
}