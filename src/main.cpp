
#include <iostream>
#include "support/config.hpp"
#include "SysYLexer.h"
#include "visitor/visitor.hpp"
#include "pass/pass.hpp"

#include "mir/MIR.hpp"
#include "mir/target.hpp"
#include "mir/lowering.hpp"
#include "target/riscv/RISCV.hpp"
#include "target/riscv/RISCVTarget.hpp"

#include "support/FileSystem.hpp"
#include "support/Profiler.hpp"

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
  sysy::SysYIRGenerator gen(&module, ast_root);
  {
    utils::Stage stage("IR Generation"sv);
    gen.build_ir();
  }
  if (not module.verify(std::cerr)) {
    module.print(std::cerr);
    std::cerr << "IR verification failed" << std::endl;
    // assert(false && "IR verification failed");
  }

  //! 2. Optimization Passes
  auto tAIM = new pass::TopAnalysisInfoManager(&module);
  tAIM->initialize();
  auto pm = new pass::PassManager(&module, tAIM);
  {
    utils::Stage stage("Optimization Passes"sv);
    pm->runPasses(config.passes);
  }

  for (auto pass : config.passes) {
    std::cerr << "Pass: " << pass << std::endl;
  }

  if (config.genIR) {  // ir print
    if (config.outfile.empty()) {
      module.print(std::cout);
    } else {
      ofstream fout;
      fout.open(config.outfile);
      module.print(fout);
    }
  }

  //! 3. Code Generation

  if (config.genASM) {
    auto target = mir::RISCVTarget();
    auto mir_module = mir::createMIRModule(module, target, tAIM);
    if (config.outfile.empty()) {
      target.emit_assembly(std::cout, *mir_module);
    } else {
      ofstream fout;
      fout.open(config.outfile);
      target.emit_assembly(fout, *mir_module);
    }
    {
      auto filename = utils::preName(config.infile) + ".s";
      auto path = config.debugDir() / filename;
      ofstream fout(path);
      target.emit_assembly(fout, *mir_module);
    }
  }

  if (config.logLevel >= sysy::LogLevel::DEBUG) {
    utils::Profiler::get().printStatistics();
  }
  return EXIT_SUCCESS;
}