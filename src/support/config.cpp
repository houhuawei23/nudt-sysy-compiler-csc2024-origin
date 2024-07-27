#include "support/config.hpp"

#include <cstring>
#include <getopt.h>
#include <string_view>
using namespace std::string_view_literals;

namespace sysy {

/*
-i: Generate IR
-t {passname} {pasename} ...: opt passes names to run
-o {filename}:  output file, default gen.ll (-ir) or gen.s (-S)
-S: gen assembly
-O[0-3]: opt level

./compiler-f test.c -i -t mem2reg dce -o gen.ll
./compiler -f test.c -i -t mem2reg -o gen.ll -O0 -L0
*/

std::string_view HELP = R"(
Usage: ./compiler [options]
  -f {filename}         input file
  -i                    Generate IR
  -t {passname} ...     opt passes names to run
  -o {filename}         output file, default gen.ll (-ir) or gen.s (-S)
  -S                    gen assembly
  -O[0-3]               opt level
  -L[0-2]               log level: 0=SILENT, 1=INFO, 2=DEBUG

Examples:
$ ./compiler -f test.c -i -t mem2reg -o gen.ll -O0 -L0
$ ./compiler -f test.c -i -t mem2reg dce -o gen.ll
)";

void Config::print_help() {
  std::cout << HELP << std::endl;
}

void Config::print_info() {
  if (logLevel > LogLevel::SILENT) {
    std::cout << "In File  : " << infile << std::endl;
    std::cout << "Out File : " << outfile << std::endl;
    std::cout << "Gen IR   : " << (genIR ? "Yes" : "No") << std::endl;
    std::cout << "Gen ASM  : " << (genASM ? "Yes" : "No") << std::endl;
    std::cout << "Opt Level: " << optLevel << std::endl;
    std::cout << "Log Level: " << logLevel << std::endl;
    if (not passes.empty()) {
      std::cout << "Passes   : ";
      for (const auto& pass : passes) {
        std::cout << " " << pass;
      }
      std::cout << std::endl;
    }
  }
}

void Config::parseTestArgs(int argc, char* argv[]) {
  int option;
  while ((option = getopt(argc, argv, "f:it:o:SO:L:")) != -1) {
    switch (option) {
      case 'f':
        infile = optarg;
        break;
      case 'i':
        genIR = true;
        break;
      case 't':
        // optind start from 1, so we need to minus 1
        while (optind <= argc && *argv[optind - 1] != '-') {
          passes.push_back(argv[optind - 1]);
          optind++;
        }
        optind--;  // must!
        break;
      case 'o':
        outfile = optarg;
        break;
      case 'S':
        genASM = true;
        break;
      case 'O':
        optLevel = static_cast<OptLevel>(std::stoi(optarg));
        break;
      case 'L':
        logLevel = static_cast<LogLevel>(std::stoi(optarg));
        break;
      default:
        print_help();
        exit(EXIT_FAILURE);
    }
  }
}

// clang-format off
static const auto PassesList = std::vector<std::string>{
  "mem2reg",  //
  "simplifycfg", //error in backend CFGAnalysis.successors
  // "instcombine",  //
  "adce",     // passed all functional
  "inline",   // segfault on 60_sort_test6/69_expr_eval/...
  "tco",      // tail call optimization
  "inline",   //
  "g2l",      // global to local
  "instcombine",  //
  "adce",     // passed all functional
  "sccp",     //
  "gcm",  // global code motion
  "gvn",          // global value numbering: passed, slow
  "instcombine",  //
  "adce",         //
  "simplifycfg",
  "reg2mem"
};


// static const auto PassesList = std::vector<std::string>{
//   "mem2reg",  //
//   "simplifycfg", //  error in backend CFGAnalysis.successors
//   // "test",
//   "reg2mem"
//   // "inline"
// };
// clang-format on
/*
5
功能测试：compiler -S -o testcase.s testcase.sy
6
性能测试：compiler -S -o testcase.s testcase.sy -O1
7
debug: compiler -S -o testcase.s testcase.sy -O1 -L2
*/
void Config::parseSubmitArgs(int argc, char* argv[]) {
  genASM = true;
  outfile = argv[3];
  infile = argv[4];
  optLevel = OptLevel::O1;

  if (argc == 7) {
    if (argv[6] == "-L2"sv) {
      logLevel = LogLevel::DEBUG;
    }
  }

  /* 性能测试 */
  // if (argc == 6) {
  //   optLevel = OptLevel::O1;
  //   // std::cerr << "using default opt level -O1" << std::endl;
  // }
}

void Config::parseCmdArgs(int argc, char* argv[]) {
  if (argv[1] == "-f"sv) {
    parseTestArgs(argc, argv);
  } else if (argv[1] == "-S"sv) {
    parseSubmitArgs(argc, argv);
  } else {
    print_help();
    exit(EXIT_FAILURE);
  }
  if (optLevel == OptLevel::O1) {
    passes = PassesList;
  }
}

}  // namespace sysy
