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
        unsigned int DeadallocaNum = 0;
        unsigned int SinglestoreNum = 0;
        ir::StoreInst *OnlyStore;
        ir::BasicBlock *OnlyBlock;
        std::vector<ir::AllocaInst *> Allocas;
        std::map<ir::AllocaInst *, std::vector<ir::BasicBlock *>> DefsBlock;
        std::map<ir::AllocaInst *, std::vector<ir::BasicBlock *>> UsesBlock;

        std::map<ir::BasicBlock *, std::map<ir::PhiInst *, ir::AllocaInst *>> PhiMap;
        std::map<ir::AllocaInst *, ir::Argument *> ValueMap;

    public:
        void run(ir::Function *func) override;
        std::string name() override
        {
            return "mem2reg";
        }
        void promotememToreg(ir::Function *F);
        void RemoveFromAllocasList(unsigned &AllocaIdx);
        void allocaAnalysis(ir::AllocaInst *alloca);
        bool promotemem2reg(ir::Function *F);
        bool is_promoted(ir::AllocaInst *alloca);
        bool rewriteSingleStoreAlloca(ir::AllocaInst *alloca);
        int getStoreinstindexinBB(ir::BasicBlock *BB, ir::StoreInst *I);
        int getLoadeinstindexinBB(ir::BasicBlock *BB, ir::LoadInst *I);
    };
} // namespace pass
