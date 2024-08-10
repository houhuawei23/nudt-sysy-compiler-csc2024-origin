#include "pass/analysis/dependenceAnalysis/DependenceAnalysis.hpp"
#include "pass/analysis/dependenceAnalysis/dpaUtils.hpp"
using namespace pass;


/*
需要满足进行依赖关系分析的条件和建议
1. 在gvn licm gcm dle dse之后调用
*/
/*
如何进行依赖关系分析？
我们在sysy的语法条件下可以对一般的依赖关系分析框架进行一些简化
划分为了三层框架:
1. 基地址
2. 带idx的地址
3. memWR
我们知道，要产生依赖，就必须保证两次内存操作的地址一致，这个一致包括了两个层面的一致
1. 基地址的一致(可以理解为数组首地址)
2. idx地址的一致(也就是索引一致，或者可能一致)
基地址，也就是数组的首地址，分为三类
1. global
2. local 
3. arg
他们分别代表数组首地址的三种来源
我们可以简单的根据其种类对这些数组首地址进行别名分析，（由其是arg类的）根据
当前函数被调用时候的情况判断这些arg是否和global或是外层的local数组一致
样例中没有，因此不考虑将一个多维数组降维（不完全解引用）传入的情况

子地址，带idx的地址，实际上是基地址经过gep语句生成的地址，我们要维护从基地址
到gep的映射，看看一个基地址有哪些索引衍生出的地址
这些地址之间的关系，也就是我们要分析的依赖关系了
值得注意的是，我们无需维护在循环体中这些对内存存取的先后顺序

在收集了这些信息的基础上，就可以针对某一个baseaddr的子地址们进行ZIV SIV等测试了
判断他们是否可能产生依赖
*/

void dependenceAnalysis::run(ir::Function* func,TopAnalysisInfoManager* tp){
    if(func->isOnlyDeclare())return;
    domctx=tp->getDomTree(func);
    lpctx=tp->getLoopInfo(func);
    idvctx=tp->getIndVarInfo(func);
    sectx=tp->getSideEffectInfo();
    dpctx=tp->getDepInfoWithoutRefresh(func);
    for(auto lp:lpctx->loops()){//按照循环树来遍历处理各个循环，总是先从内层循环开始分析
        if(lp->parent()!=nullptr)continue;
        runOnLoop(lp,tp);
    }
}

void dependenceAnalysis::runOnLoop(ir::Loop* lp,TopAnalysisInfoManager* tp){
    for(auto subLoop:lp->subLoops()){
        runOnLoop(subLoop,tp);
    }
    auto depInfo=dpctx->dpInfo(lp);
    //先处理子循环再处理自己
    if(lp->subLoops().empty()){//如果就是一个独立的循环
        for(auto bb:lp->blocks()){
            for(auto inst:bb->insts()){
                memRW* pMemRW;
                ir::Value* baseptr;
                if(auto ldInst=inst->dynCast<ir::LoadInst>()){
                    auto ptr=ldInst->ptr();
                    baseptr=addrToBaseaddr(ptr);
                    pMemRW=makeMemRW(ptr,memread,ldInst);
                }
                else if(auto stInst=inst->dynCast<ir::StoreInst>()){
                    auto ptr=stInst->ptr();
                    baseptr=addrToBaseaddr(ptr);
                    pMemRW=makeMemRW(ptr,memwrite,stInst);
                }
                depInfo->addMemRW(pMemRW,baseptr);
            }
        }
        
    }
    else{//如果有嵌套的循环
        auto lpDepInfo=dpctx->dpInfo(lp);
        for(auto sbLp:lp->subLoops()){
            auto sbLpDepInfo=dpctx->dpInfo(sbLp);
            for(auto bd:sbLpDepInfo->baseAddrs){
                lpDepInfo->baseAddrs.insert(bd);

            }
        }
    }
}

//接受指针和读写来创造memWR
memRW* dependenceAnalysis::makeMemRW(ir::Value* ptr,memOp memop,ir::Instruction* inst){
    auto pnewMemRW=new memRW;
    pnewMemRW->memop=memop;
    pnewMemRW->inst=inst;
    assert(ptr->dynCast<ir::GetElementPtrInst>()!=nullptr);
    pnewMemRW->gepPtr=ptr->dynCast<ir::GetElementPtrInst>();
    return pnewMemRW;
}

void dependenceAnalysisInfoCheck::run(ir::Function* func,TopAnalysisInfoManager* tp){
    dpctx=tp->getDepInfoWithoutRefresh(func);
    lpctx=tp->getLoopInfoWithoutRefresh(func);
    std::cerr<<"In function \""<<func->name()<<"\":"<<std::endl;
    for(auto lp:lpctx->loops()){
        auto depinfo=dpctx->dpInfo(lp);
        depinfo->print(std::cerr);
    }
}

subAddrIdx* dependenceAnalysis::getSubAddrIdx(ir::GetElementPtrInst* gep,ir::Loop* lp,ir::Value* baseptr){
    auto idv=idvctx->getIndvar(lp);
    auto pnewSubAddrIdx=new subAddrIdx;
    auto& subAddrIdxListVec=pnewSubAddrIdx->idxlist;
    auto& subAddrIdxTypeListVec=pnewSubAddrIdx->idxTypes;
    auto curPtr=gep;
    while(curPtr!=baseptr){
        auto curIdx=curPtr->index();
        subAddrIdxListVec.push_back(curIdx);
        subAddrIdxTypeListVec[curIdx]=getIdxType(curIdx,lp,idv);
        curPtr=curPtr->value()->dynCast<ir::GetElementPtrInst>();
    }
}

idxType dependenceAnalysis::getIdxType(ir::Value* idxVal,ir::Loop* lp,ir::indVar* idv){
    if(idxVal->dynCast<ir::ConstantValue>())return iCONST;
    if(isSimplyLpInvariant(idxVal,lp))return iLPINVARIANT;
    if(idv!=nullptr){
        if(idv->phiinst()==idxVal)
            return iIDV;
        else{
            
        }
    }
}

bool dependenceAnalysis::isSimplyLpInvariant(ir::Value* val,ir::Loop* lp){
    if(val->dynCast<ir::ConstantValue>())return true;
    if(val->dynCast<ir::Argument>())return true;
    auto inst=val->dynCast<ir::Instruction>();
    if(inst==nullptr)return false;
    return domctx->dominate(inst->block(),lp->header()) and inst->block()!=lp->header();
}