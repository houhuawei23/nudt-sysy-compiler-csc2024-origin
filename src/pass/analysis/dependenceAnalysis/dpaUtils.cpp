#include "pass/analysis/dependenceAnalysis/DependenceAnalysis.hpp"
#include "pass/analysis/dependenceAnalysis/dpaUtils.hpp"
using namespace pass;

//将地址的基址提取出来
ir::Value* pass::addrToBaseaddr(ir::Value* ptr){
    if(auto ptrArg=ptr->dynCast<ir::Argument>())return ptr;
    if(auto ptrGv=ptr->dynCast<ir::GlobalVariable>())return ptr;
    if(auto ptrAlloca=ptr->dynCast<ir::AllocaInst>())return ptr;
    if(auto ptrGep=ptr->dynCast<ir::GetElementPtrInst>()){
        while(1){
            auto newPtr=ptrGep->value();
            ptrGep=newPtr->dynCast<ir::GetElementPtrInst>();
            if(ptrGep==nullptr)return newPtr;
        }
    }
    assert(false && "invalid input in function\"addrToBaseAddr\"");
    return nullptr;
}

pass::baseAddrType pass::getBaseaddrType(ir::Value* ptr){
    if(auto ptrArg=ptr->dynCast<ir::Argument>())return typearg;
    if(auto ptrGv=ptr->dynCast<ir::GlobalVariable>())return typeglobal;
    if(auto ptrAlloca=ptr->dynCast<ir::AllocaInst>())return typelocal;
    assert(false and "Invalid input for function\"getBaseaddrType\"!");
}

//返回假就是一定不一致
//不接受将部分解引用的数组首地址作为参数传入，否则将得出错误的结果！
bool pass::isBaseAddrPossiblySame(ir::Value* ptr1,ir::Value* ptr2,ir::Function* func,callGraph* cgctx){
    //必须是三种基址中的一种
    auto type1=getBaseaddrType(ptr1);
    auto type2=getBaseaddrType(ptr2);
    if(type1==type2){
        if(type1==typeglobal or type1==typelocal){
            return ptr1==ptr2;
        }
        //when they are all args
        auto arg1=ptr1->dynCast<ir::Argument>();
        auto arg2=ptr2->dynCast<ir::Argument>();
        auto idx1=arg1->index();
        auto idx2=arg2->index();    
        for(auto callInst:cgctx->calleeCallInsts(func)){
            auto rarg1=callInst->rargs()[idx1];
            auto rarg2=callInst->rargs()[idx2];
            auto baseAddr1=addrToBaseaddr(rarg1->value());
            auto baseAddr2=addrToBaseaddr(rarg2->value());
            if(isBaseAddrPossiblySame(baseAddr1,baseAddr2,callInst->block()->function(),cgctx))
                return true;
        }
        return false;

    }
    else{
        if(type1!=typearg and type2!=typearg){//如两者均不是arg,类型又不一致,就说明不一致
            return false;
        }
        if(type1==typelocal or type2==typelocal){//如果两者中有一个是Typelocal,另一个是typearg,则可以明显推断出来不一致
            return false;
        }
        //留下来的只有arg和global这种情况！
        auto gv1=ptr1->dynCast<ir::GlobalVariable>();
        auto gv2=ptr2->dynCast<ir::GlobalVariable>();
        ir::Argument* arg;
        ir::GlobalVariable* gv;
        if(gv1!=nullptr and gv2==nullptr){
            gv=gv1;
            arg=ptr2->dynCast<ir::Argument>();
        }
        else if(gv1==nullptr and gv2!=nullptr){
            gv=gv2;
            arg=ptr2->dynCast<ir::Argument>();
        }
        else
            assert(false and "error in function\"isPossiblySameBaseAddr\"") ;
        auto idx=arg->index();
        for(auto callInst:cgctx->calleeCallInsts(func)){
            auto rarg=callInst->rargs()[idx];
            if(isBaseAddrPossiblySame(gv,rarg->value(),callInst->block()->function(),cgctx))
                return true;
        }
        return false;
    }
}
