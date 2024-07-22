// clang-format off
#include <iostream>
#include "support/config.hpp"
#include "SysYLexer.h"
#include "visitor/visitor.hpp"
#include "pass/pass.hpp"


#include "mir/mir.hpp"
#include "mir/target.hpp"
#include "mir/lowering.hpp"
#include "target/riscv/RISCV.hpp"
#include "target/riscv/RISCVTarget.hpp"

#include "support/FileSystem.hpp"

namespace fs = std::filesystem;

using namespace std;
/* ./compiler -f test.c -i -t mem2reg -o gen.ll -O0 -L0 */
int main(int argc, char* argv[]) {
    auto& config = sysy::Config::getInstance();
    config.parse_cmd_args(argc, argv);
    // config.parseSubmitArgs(argc, argv);
    config.print_info();

    if (config.infile.empty()) {
        cerr << "Error: input file not specified" << endl;
        config.print_help();
        return EXIT_FAILURE;
    }
    // mkdir ./.debug/xxx/ for debug info
    if (config.log_level == sysy::LogLevel::DEBUG) {
      utils::ensure_directory_exists(config.debugDir());
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
    pm->runPasses(config.pass_names);
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
    constexpr bool DebugDomBFS = false;
    for (auto fun : module_ir->funcs()) {
        if(fun->isOnlyDeclare())continue;
        auto dom_ctx = tAIM->getDomTree(fun);
        dom_ctx->refresh();
        dom_ctx->BFSDomTreeInfoRefresh();
        auto dom_vec = dom_ctx->BFSDomTreeVector();

        if (DebugDomBFS) {
            for (auto bb : dom_ctx->BFSDomTreeVector()) {
                std::cerr << bb->name() << " ";
            }
            std::cerr << "\n";
        }
    }
    if (config.gen_asm) {
        auto target = mir::RISCVTarget();
        auto mir_module = mir::createMIRModule(*module_ir, target, tAIM);
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