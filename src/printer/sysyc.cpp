#include "ASTPrinter.h"
#include <cstdlib>
#include <fstream>
#include <iostream>
using namespace std;
using namespace antlr4;
#include "ASTPrinter.h"
#include "SysYLexer.h"
#include "SysYParser.h"

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
    ANTLRInputStream input(fin);
    SysYLexer lexer(&input);
    CommonTokenStream tokens(&lexer);
    SysYParser parser(&tokens);

    SysYParser::CompUnitContext* astrootptr = parser.compUnit();
    ASTPrinter printer;
    printer.visitCompUnit(astrootptr);

    return EXIT_SUCCESS;
}