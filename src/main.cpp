
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
/*
config.parseTestArgs:
./compiler -f test.c -i -t mem2reg -o gen.ll -O0 -L0

config.parseSubmitArgs
./compiler -S -o testcase.s testcase.sy
 */
int main(int argc, char* argv[]) {
  auto& config = sysy::Config::getInstance();

  // config.parseTestArgs(argc, argv);
  // config.parseSubmitArgs(argc, argv);
  config.parseCmdArgs(argc, argv);
  config.print_info();

  if (config.infile.empty()) {
    cerr << "Error: input file not specified" << endl;
    config.print_help();
    return EXIT_FAILURE;
  }
  // mkdir ./.debug/xxx/ for debug info
  if (config.logLevel == sysy::LogLevel::DEBUG) {
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
  auto base_module = &module;
  sysy::SysYIRGenerator gen(base_module, ast_root);
  auto module_ir = gen.build_ir();
  if (not module_ir->verify(std::cerr)) {
    std::cerr << "IR verification failed" << std::endl;
    assert(false && "IR verification failed");
  }

  //! 2. Optimization Passes
  auto tAIM = new pass::TopAnalysisInfoManager(module_ir);
  tAIM->initialize();
  auto pm = new pass::PassManager(module_ir, tAIM);

  pm->runPasses(config.passes);

  if (config.genIR) {  // ir print
    if (config.outfile.empty()) {
      module_ir->print(std::cout);
    } else {
      ofstream fout;
      fout.open(config.outfile);
      module_ir->print(fout);
    }
  }



  //! 3. Code Generation
  constexpr bool DebugDomBFS = false;
  for (auto fun : module_ir->funcs()) {
    if (fun->isOnlyDeclare()) continue;
    auto dom_ctx = tAIM->getDomTree(fun);
    // dom_ctx->setOff();
    dom_ctx->refresh();
    dom_ctx->BFSDomTreeInfoRefresh();
    auto dom_vec = dom_ctx->BFSDomTreeVector();

    if (DebugDomBFS) {
      for (auto bb : dom_ctx->BFSDomTreeVector()) {
        std::cerr << bb->name() << " "<< bb->insts().size() <<std::endl;
        for(auto bbdomson : dom_ctx->domson(bb)){
          std::cerr << bbdomson->name() << " " << bbdomson->insts().size() << " ";
        }
        std::cerr<<std::endl;
      }
      for (auto bb : fun->blocks()) {
        std::cerr << bb->name() << " "<<bb->insts().size()<<std::endl;
      }
      std::cerr << "\n";
    }
  }

  if (config.genASM) {
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
      ofstream fout("./.debug/" + utils::preName(config.infile) + ".s");
      target.emit_assembly(fout, *mir_module);
    }
  }

  return EXIT_SUCCESS;
}