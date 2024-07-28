#include "pass/optimize/DSE.hpp"

using namespace pass;

static std::unordered_map<ir::Value*,ir::StoreInst*>ptrMap;
static std::set<ir::StoreInst*>removeInsts;

void simpleDSE::run(ir::BasicBlock* bb,TopAnalysisInfoManager* tp){
    ptrMap.clear();
    removeInsts.clear();
    for(auto inst:bb->insts()){
        if(auto storeInst=dynamic_cast<ir::StoreInst*>(inst)){
            if(ptrMap.count(storeInst->ptr())){
                auto oldStoreInst=ptrMap[storeInst->ptr()];
                removeInsts.insert(oldStoreInst);
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
    if(removeInsts.size()==0)return;
    std::cerr<<"Delete "<<removeInsts.size()<<" store insts."<<std::endl;
    for(auto inst:removeInsts){
        bb->delete_inst(inst);
    }
}
