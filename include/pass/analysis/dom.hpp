#pragma once
#include "pass/pass.hpp"

namespace pass{

class domInfoPass:public FunctionPass{
    public:
        void run(ir::Function* func,topAnalysisInfoManager* tp) override;
        
};

class preProcDom:public FunctionPass{
    private:
        domTree* domctx;
    public:
        void run(ir::Function* func,topAnalysisInfoManager* tp) override;
        
};

class idomGen:public FunctionPass{
    private:
        domTree* domctx;
    private:
        void dfsBlocks(ir::BasicBlock* bb);
        ir::BasicBlock* eval(ir::BasicBlock* bb);
        void link(ir::BasicBlock* v,ir::BasicBlock* w);
        void compress(ir::BasicBlock* bb);
    public:
        void run(ir::Function* func,topAnalysisInfoManager* tp) override;
        
};

class domFrontierGen:public FunctionPass{
    private:
        domTree* domctx;
    private:
        void getDomTree(ir::Function* func);
        void getDomFrontier(ir::Function* func);
        void getDomInfo(ir::BasicBlock* bb, int level);
    public:
        void run(ir::Function* func,topAnalysisInfoManager* tp) override;
        
};

class domInfoCheck:public FunctionPass{
    private:
        domTree* domctx;
    public:
        void run(ir::Function* func,topAnalysisInfoManager* tp) override;
};

}
