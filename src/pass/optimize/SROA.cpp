#include "pass/analysis/dependenceAnalysis/DependenceAnalysis.hpp"
#include "pass/optimize/SROA.hpp"
using namespace pass;

static std::set<ir::GetElementPtrInst*>processedSubAddrs;

std::string SROA::name() const {
    return "SROA";
}

//SROA基本逻辑：
/*
处理的是对于循环内的变量
对于所有的循环内内存存取
1. 只考虑数组，全局由AG2L来考虑
2. 对于循环不变量，直接创建临时变量进行修改(从外到内，尽可能在外层做，将load-store紧贴在对应gep之后)
3. 对于当前循环内的量，只要没有possiblySame，都可以进行替换，从外到内即可
*/

void SROA::run(ir::Function* func,TopAnalysisInfoManager* tp){
    dpctx=tp->getDepInfo(func);
    domctx=tp->getDomTreeWithoutRefresh(func);
    idvctx=tp->getIndVarInfoWithoutRefresh(func);
    lpctx=tp->getLoopInfoWithoutRefresh(func);
    sectx=tp->getSideEffectInfoWithoutRefresh();
    processedSubAddrs.clear();
    for(auto lp:lpctx->loops()){
        if(lp->parentloop()!=nullptr)continue;
        runOnLoop(lp);
    }
}

void SROA::runOnLoop(ir::Loop* lp){
    depInfoForLp=dpctx->getLoopDependenceInfo(lp);
    //对当前循环一对对subAddr进行分析，对于有重复的将重复的加入到集合中并不再处理
    std::set<ir::GetElementPtrInst*>hasRepGeps;
    std::set<ir::GetElementPtrInst*>SROAGeps;
    //主要是看迭代内是否完全不一致
    for(auto bd:depInfoForLp->getBaseAddrs()){
        auto& subAddrs=depInfoForLp->baseAddrToSubAddrSet(bd);
        for(auto setIter=subAddrs.begin();setIter!=subAddrs.end();setIter++){
            //如果当前的subAddr已经被处理了，就不再处理continue
            bool isCurIterIndependent=true;
            for(auto setIter2=setIter;setIter2!=subAddrs.end();setIter2++){
                
            }
            //如果与其他的在当前循环迭代均不一样，就可以进行Sroa
            if(isCurIterIndependent) {
                // SROAGeps.insert()
            }
        }
    }
    
    //从外到内
    for(auto subLoop:lp->subLoops()){
        runOnLoop(subLoop);
    }
}

ir::Value* SROA::createNewLocal(ir::Type* allocaType,ir::Function* func){
    //创建一个新的局部变量（alloca）
    auto funcEntry=func->entry();
    auto newAlloca=new ir::AllocaInst(allocaType,false,funcEntry);
    funcEntry->emplace_lastbutone_inst(newAlloca);
    return newAlloca;
}

//对迭代内起作用,如果是真正的LpI指针,还有另外的做法
bool SROA::replaceAllUseInLpIdv(ir::GetElementPtrInst* gep,ir::Loop* lp,ir::AllocaInst* newAlloca,bool isOnlyRead,bool isOnlyWrite){
    auto gepBB=gep->block();
    auto gepPos=std::find(gepBB->insts().begin(),gepBB->insts().end(),gep);
    gepPos++;
    if(isOnlyRead){
        //将gep load出来,store到alloca中，然后将所有的lp中的load替代之
        for(auto puseIter=gep->uses().begin();puseIter!=gep->uses().end();){
            auto puse=*puseIter;
            puseIter++;
            auto puser=puse->user();
            auto puseIdx=puse->index();
            auto userInst=puser->dynCast<ir::Instruction>();
            if(lp->blocks().count(userInst->block())){
                userInst->setOperand(puseIdx,newAlloca);
            }
        }
        auto gepLoad=new ir::LoadInst(gep,gep->baseType(),gepBB);
        auto storeToAlloca=new ir::StoreInst(gepLoad,newAlloca,gepBB);
        gepBB->emplace_inst(gepPos,gepLoad);
        gepBB->emplace_inst(gepPos,storeToAlloca);
        return true;
    }
    else if(isOnlyWrite){
        auto lpLatch=*lp->latchs().begin();
        if(lp->latchs().size()>1)return false;
        //直接将所有lp中的store替换成给alloca store，最后load一下alloca，然后store到gep（循环迭代末尾）
        for(auto puseIter=gep->uses().begin();puseIter!=gep->uses().end();){
            auto puse=*puseIter;
            puseIter++;
            auto puser=puse->user();
            auto puseIdx=puse->index();
            auto userInst=puser->dynCast<ir::Instruction>();
            if(lp->blocks().count(userInst->block())){
                userInst->setOperand(puseIdx,newAlloca);
            }
        }
        
        auto loadAlloca=new ir::LoadInst(newAlloca,newAlloca->baseType(),lpLatch);
        auto storeLoadToGep=new ir::StoreInst(loadAlloca,gep,lpLatch);
        lpLatch->emplace_first_inst(loadAlloca);
        lpLatch->emplace_first_inst(storeLoadToGep);
        return true;
    }
    else{
        auto lpLatch=*lp->latchs().begin();
        if(lp->latchs().size()>1)return false;
        //将所有对其的读写变为对alloca的读写
        for(auto puseIter=gep->uses().begin();puseIter!=gep->uses().end();){
            auto puse=*puseIter;
            puseIter++;
            auto puser=puse->user();
            auto puseIdx=puse->index();
            auto userInst=puser->dynCast<ir::Instruction>();
            if(lp->blocks().count(userInst->block())){
                userInst->setOperand(puseIdx,newAlloca);
            }
        }
        //将gep load出来,store到alloca中
        auto gepLoad=new ir::LoadInst(gep,gep->baseType(),gepBB);
        auto storeToAlloca=new ir::StoreInst(gepLoad,newAlloca,gepBB);
        gepBB->emplace_inst(gepPos,gepLoad);
        gepBB->emplace_inst(gepPos,storeToAlloca);
        //将alloca load出来，将其值store进去
        
        auto loadAlloca=new ir::LoadInst(newAlloca,newAlloca->baseType(),lpLatch);
        auto storeLoadToGep=new ir::StoreInst(loadAlloca,gep,lpLatch);
        lpLatch->emplace_first_inst(loadAlloca);
        lpLatch->emplace_first_inst(storeLoadToGep);
        return true;
    }
}

bool SROA::replaceAllUseInLpForLpI(ir::GetElementPtrInst* gep,ir::Loop* lp,ir::AllocaInst* newAlloca,bool isOnlyRead,bool isOnlyWrite){
    auto gepBB=gep->block();
    auto gepPos=std::find(gepBB->insts().begin(),gepBB->insts().end(),gep);
    gepPos++;
        if(isOnlyRead){
        //将gep load出来,store到alloca中，然后将所有的lp中的load替代之
        for(auto puseIter=gep->uses().begin();puseIter!=gep->uses().end();){
            auto puse=*puseIter;
            puseIter++;
            auto puser=puse->user();
            auto puseIdx=puse->index();
            auto userInst=puser->dynCast<ir::Instruction>();
            if(lp->blocks().count(userInst->block())){
                userInst->setOperand(puseIdx,newAlloca);
            }
        }
        auto gepLoad=new ir::LoadInst(gep,gep->baseType(),gepBB);
        auto storeToAlloca=new ir::StoreInst(gepLoad,newAlloca,gepBB);
        gepBB->emplace_inst(gepPos,gepLoad);
        gepBB->emplace_inst(gepPos,storeToAlloca);
        return true;
    }
    else if(isOnlyWrite){
        auto lpExit=*lp->exits().begin();
        if(lp->exits().size()>1)return false;
        //直接将所有lp中的store替换成给alloca store，最后load一下alloca，然后store到gep（循环迭代末尾）
        for(auto puseIter=gep->uses().begin();puseIter!=gep->uses().end();){
            auto puse=*puseIter;
            puseIter++;
            auto puser=puse->user();
            auto puseIdx=puse->index();
            auto userInst=puser->dynCast<ir::Instruction>();
            if(lp->blocks().count(userInst->block())){
                userInst->setOperand(puseIdx,newAlloca);
            }
        }
        
        auto loadAlloca=new ir::LoadInst(newAlloca,newAlloca->baseType(),lpExit);
        auto storeLoadToGep=new ir::StoreInst(loadAlloca,gep,lpExit);
        lpExit->emplace_first_inst(loadAlloca);
        lpExit->emplace_first_inst(storeLoadToGep);
        return true;
    }
    else{
        auto lpExit=*lp->exits().begin();
        if(lp->exits().size()>1)return false;
        //将所有对其的读写变为对alloca的读写
        for(auto puseIter=gep->uses().begin();puseIter!=gep->uses().end();){
            auto puse=*puseIter;
            puseIter++;
            auto puser=puse->user();
            auto puseIdx=puse->index();
            auto userInst=puser->dynCast<ir::Instruction>();
            if(lp->blocks().count(userInst->block())){
                userInst->setOperand(puseIdx,newAlloca);
            }
        }
        //将gep load出来,store到alloca中
        auto gepLoad=new ir::LoadInst(gep,gep->baseType(),gepBB);
        auto storeToAlloca=new ir::StoreInst(gepLoad,newAlloca,gepBB);
        gepBB->emplace_inst(gepPos,gepLoad);
        gepBB->emplace_inst(gepPos,storeToAlloca);
        //将alloca load出来，将其值store进去
        auto loadAlloca=new ir::LoadInst(newAlloca,newAlloca->baseType(),lpExit);
        auto storeLoadToGep=new ir::StoreInst(loadAlloca,gep,lpExit);
        lpExit->emplace_first_inst(loadAlloca);
        lpExit->emplace_first_inst(storeLoadToGep);
        return true;
    }
}