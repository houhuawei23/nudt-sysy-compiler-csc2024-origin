#pragma once
#include <set>
#include <cassert>
#include <map>
#include <vector>
#include "ir/ir.hpp"
#include "pass/pass.hpp"

namespace pass
{
    class Mem2Reg : public FunctionPass
    {
    private:
        domTree *domctx;
        // unsigned int SinglestoreNum = 0;
        // ir::StoreInst *OnlyStore;
        // ir::BasicBlock *OnlyBlock;
        // bool OnlyUsedInOneBlock;
        std::vector<ir::AllocaInst *> Allocas;
        std::map<ir::AllocaInst *, std::set<ir::BasicBlock *>> DefsBlock;
        std::map<ir::AllocaInst *, std::set<ir::BasicBlock *>> UsesBlock;
        // std::map<ir::AllocaInst *, std::vector<ir::BasicBlock *>> DefsBlockvector;
        // std::map<ir::AllocaInst *, std::vector<ir::BasicBlock *>> UsesBlockvector;

        std::map<ir::BasicBlock *, std::map<ir::PhiInst *, ir::AllocaInst *>> PhiMap;
        std::map<ir::AllocaInst *, ir::Argument *> ValueMap;

    public:
        void run(ir::Function *func, topAnalysisInfoManager* tp) override;
        // int getStoreNuminBB(ir::BasicBlock *BB, ir::AllocaInst *AI);
        // ir::StoreInst *getLastStoreinBB(ir::BasicBlock *BB, ir::AllocaInst *AI);
        // bool rewriteSingleStoreAlloca(ir::AllocaInst *alloca);
        // bool pormoteSingleBlockAlloca(ir::AllocaInst *alloca);
        void promotememToreg(ir::Function *F);
        void RemoveFromAllocasList(unsigned &AllocaIdx);
        void allocaAnalysis(ir::AllocaInst *alloca);
        bool promotemem2reg(ir::Function *F);
        bool is_promoted(ir::AllocaInst *alloca);
        void insertphi();
        void rename(ir::Function *F);
        // int getStoreinstindexinBB(ir::BasicBlock *BB, ir::StoreInst *I);
        // int getLoadeinstindexinBB(ir::BasicBlock *BB, ir::LoadInst *I);
    };
} // namespace pass
