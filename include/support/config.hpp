#pragma once
#include <memory>
#include <string>
#include <iostream>
#include <vector>

namespace sysy {

enum OptLevel : uint32_t { O0 = 0, O1 = 1, O2 = 2, O3 = 3 };
enum LogLevel : uint32_t { SILENT, INFO, DEBUG };

class Config {
   protected:
    std::ostream* _os;
    std::ostream* _err_os;

   public:
    std::string infile;
    std::string outfile;

    std::vector<std::string> pass_names;
    bool gen_ir = false;
    bool gen_asm = false;

    OptLevel opt_level = OptLevel::O0;
    LogLevel log_level = LogLevel::SILENT;
    
   public:
    Config() : _os(&std::cout), _err_os(&std::cerr) {}
    Config(int argc, char* argv[]) { parse_cmd_args(argc, argv); }
    ~Config() = default;

    void parse_cmd_args(int argc, char* argv[]);
    void parseSubmitArgs(int argc, char* argv[]);
    void print_help();
    void print_info();

    std::ostream& os() const { return *_os; }
    void set_ostream(std::ostream& os) { _os = &os; }
};

}  // namespace sysy
