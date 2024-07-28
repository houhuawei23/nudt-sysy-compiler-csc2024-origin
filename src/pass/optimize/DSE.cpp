#include "pass/optimize/DSE.hpp"

using namespace pass;

static std::unordered_map<ir::Value*,ir::StoreInst*>ptrMap;

void simpleDSE::run(ir::BasicBlock* bb,TopAnalysisInfoManager* tp){
    ptrMap.clear();
    for(auto inst:bb->insts()){
        if(auto storeInst=dynamic_cast<ir::StoreInst*>(inst)){
            if(ptrMap.count(storeInst->ptr())){
                auto oldStoreInst=ptrMap[storeInst->ptr()];
                bb->delete_inst(oldStoreInst);
                ptrMap[storeInst->ptr()]=storeInst;
            }
            else{
                ptrMap[storeInst->ptr()]=storeInst;
            }
        }
        else if(auto loadInst=dynamic_cast<ir::LoadInst*>(inst)){
            ptrMap.erase(loadInst->ptr());
        }
    }
}
