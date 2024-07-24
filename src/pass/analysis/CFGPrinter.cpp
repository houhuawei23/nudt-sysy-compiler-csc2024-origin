#include "pass/analysis/CFGPrinter.hpp"
using namespace pass;

void CFGPrinter::run(ir::Function* func, topAnalysisInfoManager* tp){
    using namespace std;
    func->rename();
    for(auto bb:func->blocks()){
        for(auto bbnext:bb->next_blocks()){
            cerr<<bb->name()<<" "<<bbnext->name()<<endl;
        }
    }
} 