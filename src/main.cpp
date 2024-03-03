#include "SysYLexer.h"
#include "visitor.hpp"

using namespace std;

int
main(int argc, char** argv)
{
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

    SysYParser::CompUnitContext* astrootptr = parser.compUnit();

    // ir::Module baseModule;
    sysy::SysYIRGenerator gen;
    gen.visitCompUnit(astrootptr);

    return EXIT_SUCCESS;
}