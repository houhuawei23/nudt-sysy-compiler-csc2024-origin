#pragma once
#include "pass/pass.hpp"

namespace pass{

class idomGen:public FunctionPass{
    private:
        void getDFSnum(ir::Function* func);
        void dfsBlocks(ir::BasicBlock* bb);
    public:
        void run(ir::Function* func) override;
        std::string name() override;
};



}
