#include "pass/optimize/DLE.hpp"
using namespace pass;

static::std::unordered_map<ir::Value*,ir::LoadInst*>loadedPtrSet;

void simpleDLE::run(ir::BasicBlock* bb,TopAnalysisInfoManager* tp){
    loadedPtrSet.clear();
    for(auto inst:bb->insts()){
        if(auto loadInst=dyn_cast<ir::LoadInst>(inst)){
            if(loadedPtrSet.count(loadInst->ptr())){
                auto oldLoadInst=loadedPtrSet[loadInst->ptr()];
                loadInst->replaceAllUseWith(oldLoadInst);
                bb->delete_inst(loadInst);
            }
            else{
                loadedPtrSet[loadInst->ptr()]=loadInst;
            }
        }
        else if(auto storeInst=dyn_cast<ir::StoreInst>(inst)){
            loadedPtrSet.erase(storeInst->ptr());
        }
    }
}