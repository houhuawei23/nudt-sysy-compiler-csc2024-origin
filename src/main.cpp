#include "SysYLexer.h"
#include "visitor.hpp"
#include <iostream>
using namespace std;

int main(int argc, char **argv) {
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

    SysYParser::CompUnitContext *ast_root = parser.compUnit();

    ir::Module *base_module = new ir::Module();
    sysy::SysYIRGenerator gen(base_module, ast_root); // forget to pass module
    gen.build_ir();

    auto module_ir = gen.module();
    bool genir = true;
    if (genir) {
        module_ir->print(std::cout);
    }
    return EXIT_SUCCESS;
}