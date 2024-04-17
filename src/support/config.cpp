#include "support/config.hpp"

#include <cstring>
#include <getopt.h>

namespace sysy {

/*
-i: Generate IR
-t {passname} {pasename} ...: opt passes names to run
-o {filename}:  output file, default gen.ll (-ir) or gen.s (-S)
-S: gen assembly
-O[0-3]: opt level

./main -f test.c -i -t mem2reg dce -o gen.ll
./main -f test.c -i -t mem2reg -o gen.ll -O0 -L0
*/

std::string_view HELP = R"(
Usage: ./main [options]
  -f {filename}         input file
  -i                    Generate IR
  -t {passname} ...     opt passes names to run
  -o {filename}         output file, default gen.ll (-ir) or gen.s (-S)
  -S                    gen assembly
  -O[0-3]               opt level
  -L[0-2]               log level: 0=SILENT, 1=INFO, 2=DEBUG

Examples:
$ ./main -f test.c -i -t mem2reg -o gen.ll -O0 -L0
$ ./main -f test.c -i -t mem2reg dce -o gen.ll
)";

void Config::print_help() {
    std::cout << HELP << std::endl;
}

void Config::print_info() {
    if (log_level > LogLevel::SILENT) {
        std::cout << "In       : " << infile << std::endl;
        if (outfile.empty()) {
            std::cout << "Out      : "
                      << "std::cout" << std::endl;
            // std::cout << "Out      : ";
            // if (gen_ir) {
            //     std::cout << "gen.ll" << std::endl;
            // } else {
            //     std::cout << "gen.s" << std::endl;
            // }
        } else {
            std::cout << "Out      : " << outfile << std::endl;
        }

        std::cout << "Gen IR   : " << (gen_ir ? "Yes" : "No") << std::endl;

        std::cout << "Gen ASM  : " << (gen_asm ? "Yes" : "No") << std::endl;

        std::cout << "Opt Level: " << opt_level << std::endl;

        std::cout << "Log Level: " << log_level << std::endl;

        if (not pass_names.empty()) {
            std::cout << "Passes   : ";
            for (const auto& pass : pass_names) {
                std::cout << " " << pass;
            }
            std::cout << std::endl;
        }
    }
}

void Config::parse_cmd_args(int argc, char* argv[]) {
    int option;
    while ((option = getopt(argc, argv, "f:it:o:SO:L:")) != -1) {
        switch (option) {
            case 'f':
                infile = optarg;
                break;
            case 'i':
                gen_ir = true;
                break;
            case 't':
                // optind start from 1, so we need to minus 1
                while (optind <= argc && *argv[optind - 1] != '-') {
                    pass_names.push_back(argv[optind - 1]);
                    optind++;
                }
                optind--;  // must!
                break;
            case 'o':
                outfile = optarg;
                break;
            case 'S':
                gen_asm = true;
                break;
            case 'O':
                opt_level = static_cast<OptLevel>(std::stoi(optarg));
                break;
            case 'L':
                log_level = static_cast<LogLevel>(std::stoi(optarg));
                break;
            default:
                print_help();
                exit(EXIT_FAILURE);
        }
    }
}
}  // namespace sysy