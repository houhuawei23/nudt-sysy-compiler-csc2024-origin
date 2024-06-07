#pragma once
#include "pass/pass.hpp"

namespace pass{

class preProcPostDom:public FunctionPass{
    public:
        void run(ir::Function* func) override;
        std::string name() override;
};

class ipostDomGen:public FunctionPass{
    private:
        void dfsBlocks(ir::BasicBlock* bb);
        ir::BasicBlock* eval(ir::BasicBlock* bb);
        void link(ir::BasicBlock* v,ir::BasicBlock* w);
        void compress(ir::BasicBlock* bb);
    public:
        void run(ir::Function* func) override;
        std::string name() override;
};

class postDomFrontierGen:public FunctionPass{
    private:
        void getDomTree(ir::Function* func);
        void getDomFrontier(ir::Function* func);
        void getDomInfo(ir::BasicBlock* bb, int level);
    public:
        void run(ir::Function* func) override;
        std::string name() override;
};

class postDomInfoCheck:public FunctionPass{
    public:
        void run(ir::Function* func)override;
        std::string name()override;
};

}