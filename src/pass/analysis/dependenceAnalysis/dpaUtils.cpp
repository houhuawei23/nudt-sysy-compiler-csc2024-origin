#include "pass/analysis/dependenceAnalysis/DependenceAnalysis.hpp"
#include "pass/analysis/dependenceAnalysis/dpaUtils.hpp"

using namespace pass;
//初始化lpDepInfo
void LoopDependenceInfo::makeLoopDepInfo(ir::Loop* lp){
    parent=lp;
    //遍历所有语句
    for(auto bb:lp->blocks()){
        for(auto inst:bb->insts()){
            ir::Value* baseAddr;
            if(auto ldInst=inst->dynCast<ir::LoadInst>()){
                addPtr(ldInst->ptr(),ldInst);
            }
            else if(auto stInst=inst->dynCast<ir::StoreInst>()){
                addPtr(stInst->ptr(),stInst);
            }
        }
    }

}

//取出基址
ir::Value* pass::getBaseAddr(ir::Value* val){
    if(auto gep=val->dynCast<ir::GetElementPtrInst>()){
        auto gepBaseAddr=gep->value()->dynCast<ir::GetElementPtrInst>();
        auto curGep=gep;
        while(gepBaseAddr!=nullptr){
            curGep=gepBaseAddr;
            gepBaseAddr=curGep->dynCast<ir::GetElementPtrInst>();
        }
        return curGep->value();
    }
    else{
        if(val->dynCast<ir::GlobalVariable>())return val;
        if(val->dynCast<ir::Argument>())return val;
        if(val->dynCast<ir::AllocaInst>())return val;
    }
    assert("Error occur in function \"getBaseAddr\"!" and false);
    return nullptr;
    
}

//取出基址的类型
baseAddrType pass::getBaseAddrType(ir::Value* val){
    if(val->dynCast<ir::GlobalVariable>())
        return globalType;
    if(val->dynCast<ir::Argument>())
        return argType;
    if(val->dynCast<ir::AllocaInst>())
        return localType;
    assert(false and "invalid input in function \"getBaseAddrTtype\"!");
}

//加入单个ptr的接口
void LoopDependenceInfo::addPtr(ir::Value* ptr,ir::Instruction* inst){
    auto baseAddr=getBaseAddr(ptr);
    auto subAddr=ptr->dynCast<ir::GetElementPtrInst>();
    if(subAddr==nullptr)return;//这实际上排除全局变量
    if(inst->dynCast<ir::LoadInst>()!=nullptr or inst->dynCast<ir::StoreInst>()!=nullptr)return;
    //基址
    if(baseAddrs.count(baseAddr)==0){
        baseAddrIsRead[baseAddr]=false;
        baseAddrIsWrite[baseAddr]=false;
    }
    baseAddrs.insert(baseAddr);
    if(inst->valueId()==ir::vLOAD){
        baseAddrIsRead[baseAddr]=true;
    }
    else if(inst->valueId()==ir::vSTORE){
        baseAddrIsWrite[baseAddr]=true;
    }
    else{
        assert(false and "error in function\"addPtr\", invalid input inst type!");
    }
    //子地址
    if(baseAddrToSubAddrs[baseAddr].count(subAddr)==0){
        subAddrIsRead[subAddr]=false;
        subAddrIsRead[subAddr]=false;
    }
    baseAddrToSubAddrs[baseAddr].insert(subAddr);
    if(inst->valueId()==ir::vLOAD){
        subAddrIsRead[subAddr]=true;
    }
    else if(inst->valueId()==ir::vSTORE){
        subAddrIsWrite[subAddr]=true;
    }
    else{
        assert(false and "error in function\"addPtr\", invalid input inst type!");
    }
    if(subAddrToGepIdx.count(subAddr)==0){
        auto pnewGepIdx=new gepIdx;
        while(subAddr!=nullptr){
            pnewGepIdx->idxList.push_back(subAddr->index());
            subAddr=subAddr->value()->dynCast<ir::GetElementPtrInst>();
        }
        std::reverse(pnewGepIdx->idxList.begin(),pnewGepIdx->idxList.end());
        subAddrToGepIdx[subAddr]=pnewGepIdx;
    }
    //具体语句
    subAddrToInst[subAddr].insert(inst);
    memInsts.insert(inst);
}

void LoopDependenceInfo::clearAll(){
    baseAddrs.clear();
    baseAddrToSubAddrs.clear();
    subAddrToGepIdx.clear();
    subAddrToInst.clear();
    memInsts.clear();
}

bool pass::isTwoBaseAddrPossiblySame(ir::Value* ptr1,ir::Value* ptr2){
    auto type1=getBaseAddrType(ptr1);
    auto type2=getBaseAddrType(ptr2);
    if(type1==type2){
        if(type1==globalType){
            return ptr1==ptr2;
        }
        else if(type1==localType){
            return ptr1==ptr2;
        }
        else{//分辨两个arg是否一致

        }
    }
    else{
        if(type1!=argType and type2!=argType){

        }
        else{
            if(type1==argType){

            }
            else{

            }
        }
    }   
}