#include "pass/analysis/dependenceAnalysis/DependenceAnalysis.hpp"
#include "pass/analysis/dependenceAnalysis/dpaUtils.hpp"
using namespace pass;

void dependenceAnalysis::run(ir::Function* func,TopAnalysisInfoManager* tp){
    domctx=tp->getDomTree(func);
    lpctx=tp->getLoopInfo(func);
    idvctx=tp->getIndVarInfo(func);
    sectx=tp->getSideEffectInfo();
    cgctx=tp->getCallGraph();
    dpctx=tp->getDepInfoWithoutRefresh(func);

    for(auto lp:lpctx->loops()){//从最上层的循环开始遍历
        if(lp->parentloop()==nullptr)
            runOnLoop(lp);
    }
}

void dependenceAnalysis::runOnLoop(ir::Loop* lp){
    // std::cerr<<"Here"<<std::endl;
    for(auto subLp:lp->subLoops()){//先处理子循环
        runOnLoop(subLp);
    }
    auto depInfoForLpPtr=dpctx->getLoopDependenceInfo(lp);
    if(depInfoForLpPtr==nullptr){//取出dpctx中的LoopDependenceInfo
        depInfoForLpPtr=new LoopDependenceInfo();
        dpctx->setDepInfoLp(lp,depInfoForLpPtr);
    }
    auto depInfoForLp=(LoopDependenceInfo*)depInfoForLpPtr;
    auto func=lp->header()->function();
    for(auto subLp:lp->subLoops()){//将子循环的信息合并到上层
        auto subLoopDepInfo=(LoopDependenceInfo*)(dpctx->getLoopDependenceInfo(subLp));
        depInfoForLp->getInfoFromSubLoop(subLp,subLoopDepInfo);
    }
    //分析所有的inst
    depInfoForLp->makeLoopDepInfo(lp);
    //别名分析测试
    bool isSame=false;
    for(auto setIter=depInfoForLp->getBaseAddrs().begin();setIter!=depInfoForLp->getBaseAddrs().end();setIter++){
        for(auto setIter2=depInfoForLp->getBaseAddrs().begin();setIter2!=setIter;setIter2++){
            if(setIter2==setIter)continue;
            if(isTwoBaseAddrPossiblySame(*setIter,*setIter2,func,cgctx)){
                isSame=true;
                break;
            }
        }
    }
    
    if(isSame){
        depInfoForLp->setIsBaseAddrPossiblySame(isSame);
        assert(false and "No alias is allowed in dependenceAnalysis::runOnLoop()");
    }
    //为并行设计的依赖关系分析
    // depInfoForLp->print(std::cerr);
    bool isParallel=true;
    auto defaultIdv=idvctx->getIndvar(lp);
    for(auto bd:depInfoForLp->getBaseAddrs()){
        auto& subAddrs=depInfoForLp->baseAddrToSubAddrSet(bd);
        for(auto subAd:subAddrs){
            auto gepidx=depInfoForLp->getGepIdx(subAd);
            makeGepIdx(lp,defaultIdv,gepidx);
        }
    }   
    
    // depInfoForLp->print(std::cerr);
    //进行针对并行化的依赖关系
    for(auto bd:depInfoForLp->getBaseAddrs()){
        auto& subAddrs=depInfoForLp->baseAddrToSubAddrSet(bd);
        //要么只有一个子地址，要么只有读，就说明不会有跨迭代的依赖
        if(not depInfoForLp->getIsBaseAddrWrite(bd) or subAddrs.size()==1){
            depInfoForLp->setBaseAddrIsCrossIterDep(bd,false);
        }
        
        for(auto setIter=subAddrs.begin();setIter!=subAddrs.end();setIter++){
            for(auto setIter2=subAddrs.begin();setIter2!=setIter;setIter2++){
                //在进行依赖判断的时候，自己和自己也要进行比较，确保在不同的迭代里面他们二者并不相同，如果相同就有可能产生跨循环的依赖
                auto gepidx1=depInfoForLp->getGepIdx(*setIter);
                auto gepidx2=depInfoForLp->getGepIdx(*setIter2);
                int depType=isTwoGepIdxPossiblySame(gepidx1,gepidx2,lp,defaultIdv);
                if(((depType | dCrossIterPossiblySame) !=0) or ((depType | dCrossIterTotallySame)!=0)){
                    if(depInfoForLp->getIsSubAddrWrite(*setIter) or depInfoForLp->getIsSubAddrWrite(*setIter2)){
                        isParallel=false;
                        depInfoForLp->setBaseAddrIsCrossIterDep(bd,true);
                    }
                }
            }
        }
    }
    depInfoForLp->setIsParallel(isParallel);
    depInfoForLp->print(std::cerr);
}   

void dependenceAnalysis::makeGepIdx(ir::Loop* lp,ir::indVar* idv,gepIdx* gepidx){
    for(auto val:gepidx->idxList){
        if(gepidx->idxTypes[val]!=iELSE)continue;
        if(isSimplyLoopInvariant(lp,val))gepidx->idxTypes[val]=iLOOPINVARIANT;
        if(idv!=nullptr){
            if(val==idv->phiinst())gepidx->idxTypes[val]=iIDV;
            if(isIDVPLUSMINUSFORMULA(idv,val,lp))gepidx->idxTypes[val]=iIDVPLUSMINUSFORMULA;
        }
        if(val->dynCast<ir::CallInst>())gepidx->idxTypes[val]=iCALL;
        if(val->dynCast<ir::LoadInst>())gepidx->idxTypes[val]=iLOAD;

    }
}

bool dependenceAnalysis::isSimplyLoopInvariant(ir::Loop* lp,ir::Value* val){
    if(auto constVal=val->dynCast<ir::ConstantValue>())return true;//常数
    if(auto argVal=val->dynCast<ir::Argument>())return true;//参数
    if(auto instVal=val->dynCast<ir::Instruction>()){
        return domctx->dominate(instVal->block(),lp->header()) and instVal->block()!=lp->header();
    }
    return false;
}

bool dependenceAnalysis::isIDVPLUSMINUSFORMULA(ir::indVar* idv,ir::Value* val,ir::Loop* lp){
    if(auto binaryVal=val->dynCast<ir::BinaryInst>()){
        auto lval=binaryVal->lValue();
        auto rval=binaryVal->rValue();
        if(binaryVal->valueId()!=ir::vADD and binaryVal->valueId()!=ir::vSUB)return false;
        bool isLValLpI=isSimplyLoopInvariant(lp,lval);
        bool isRValLpI=isSimplyLoopInvariant(lp,rval);
        if(not isRValLpI and not isLValLpI)return false;
        if(isRValLpI){
            if(lval==idv->phiinst())return true;
            else return isIDVPLUSMINUSFORMULA(idv,lval,lp);
        }
        else{//isLValLpI
            if(rval==idv->phiinst())return true;
            else return isIDVPLUSMINUSFORMULA(idv,rval,lp);
        }
    }
    return false;
}

int dependenceAnalysis::isTwoGepIdxPossiblySame(gepIdx* gepidx1,gepIdx* gepidx2,ir::Loop* lp,ir::indVar* idv){
    std::vector<dependenceType>compareAns;
    size_t lim=gepidx1->idxList.size();
    int res=0;
    for(size_t i=0;i<lim;i++){
        auto val1=gepidx1->idxList[i];
        auto val2=gepidx2->idxList[i];
        auto type1=gepidx1->idxTypes[val1];
        auto type2=gepidx2->idxTypes[val2];
        int outputDepInfo=isTwoIdxPossiblySame(val1,val2,type1,type2,lp,idv);
        res=res | outputDepInfo;
    }
    return res;
}

int dependenceAnalysis::isTwoIdxPossiblySame(ir::Value* val1,ir::Value* val2,idxType type1,idxType type2,ir::Loop* lp,ir::indVar* idv){
    if(val1==val2){//自己跟自己进行比较
        switch (type1)
        {
        case iLOOPINVARIANT:
            return dTotallySame | dCrossIterTotallySame;
            break;
        case iCALL:
            return dTotallySame | dCrossIterPossiblySame;
            break;
        case iIDV:
            return dTotallySame | dCrossIterTotallyNotSame;
            break;
        case iIDVPLUSMINUSFORMULA:
            return dTotallySame | dCrossIterTotallyNotSame;
            break;
        case iINNERIDV:
            return dTotallySame | dCrossIterTotallyNotSame;
            break;
        case iINNERIDVPLUSMINUSFORMULA:
            return dTotallySame | dCrossIterTotallyNotSame;
            break;
        case iLOAD:
            return dTotallyNotSame | dCrossIterPossiblySame;//TODO, 依赖于副作用等分析
            break;
        case iELSE:
            return dPossiblySame | dCrossIterPossiblySame;
            break;
        default:
            break;
        }
    }
    if(type1==type2){
        switch (type1)
        {
        case iCALL:
        {
            auto callInst1=val1->dynCast<ir::CallInst>();
            auto callInst2=val2->dynCast<ir::CallInst>();
            auto callFunc1=callInst1->callee();
            auto callFunc2=callInst2->callee();
            if(callFunc1!=callFunc2)return dPossiblySame | dCrossIterPossiblySame;
            //TODO:副作用分析判断这里是纯函数
            if(sectx->isPureFunc(callFunc1))return dTotallySame | dCrossIterPossiblySame;
            else return dPossiblySame | dCrossIterPossiblySame;
            break;
        }
            
        case iLOOPINVARIANT:
        {
            auto constval1=val1->dynCast<ir::ConstantInteger>();
            auto constval2=val2->dynCast<ir::ConstantInteger>();
            if(constval1!=nullptr and constval2!=nullptr){
                if(constval1->i32()==constval2->i32())return dTotallySame | dCrossIterTotallySame;
                else return dTotallyNotSame | dCrossIterTotallyNotSame;
            }
            else{
                return dPossiblySame | dCrossIterPossiblySame;
            }
            break;
        }
            
        case iIDV:
        {
            if(val1!=val2){
                assert(false and "Error: indvar in a same loop is not same!");
            }
            return dTotallySame | dCrossIterTotallyNotSame;
            break;
        }
           
        case iIDVPLUSMINUSFORMULA:
        {
            std::set<ir::Value*>val1Add;
            std::set<ir::Value*>val1Sub;
            auto curVal1=val1;
            while(curVal1!=idv->phiinst()){
                if(auto BInst=curVal1->dynCast<ir::BinaryInst>()){
                    auto lval=BInst->lValue();
                    auto rval=BInst->rValue();
                    ir::Value* LpIVal;
                    bool isLVAL=false;
                    if(isSimplyLoopInvariant(lp,lval)){
                        LpIVal=lval;
                        curVal1=rval;
                        isLVAL=true;
                    }
                    else if(isSimplyLoopInvariant(lp,rval)){
                        LpIVal=rval;
                        curVal1=lval;
                    }
                    else{
                        assert(false and "Error:gepIdx is not IDVPLUSMINUSFORMULA!");
                    }
                    if(BInst->valueId()==ir::vADD){
                        val1Add.insert(LpIVal);
                    }
                    else if(BInst->valueId()==ir::vSUB){
                        if(isLVAL){
                            assert(false and "Error:gepIdx is a-i formula!");
                        }
                        val1Sub.insert(LpIVal);
                    }
                }
                else{
                    assert(false and "this is not a idvplusminus formula!");
                }
            }
            auto curVal2=val2;
            while(curVal2!=idv->phiinst()){
                if(auto BInst=curVal2->dynCast<ir::BinaryInst>()){
                    auto lval=BInst->lValue();
                    auto rval=BInst->rValue();
                    ir::Value* LpIVal;
                    bool isLVAL=false;
                    if(isSimplyLoopInvariant(lp,lval)){
                        LpIVal=lval;
                        curVal1=rval;
                        isLVAL=true;
                    }
                    else if(isSimplyLoopInvariant(lp,rval)){
                        LpIVal=rval;
                        curVal1=lval;
                    }
                    else{
                        assert(false and "Error:gepIdx is not IDVPLUSMINUSFORMULA!");
                    }
                    if(BInst->valueId()==ir::vADD){
                        if(val1Add.count(LpIVal)){
                            val1Add.erase(LpIVal);
                        }   
                        else{
                            return dPossiblySame | dCrossIterPossiblySame;
                        }
                    }
                    else if(BInst->valueId()==ir::vSUB){
                        if(isLVAL){
                            assert(false and "Error:gepIdx is a-i formula!");
                        }
                        if(val1Sub.count(LpIVal)){
                            val1Sub.erase(LpIVal);
                        }   
                        else{
                            return dPossiblySame | dCrossIterPossiblySame;
                        }
                    }
                }
                else{
                    assert(false and "this is not a idvplusminus formula!");
                }
            }
            return dTotallySame | dCrossIterTotallySame;
            
            break;
        }

        case iINNERIDV:
        {
            return dPossiblySame | dCrossIterPossiblySame;
        }

        case iINNERIDVPLUSMINUSFORMULA:
        {
            return dPossiblySame | dCrossIterPossiblySame;
        }

        case iLOAD:
        {
            auto loadInst1=val1->dynCast<ir::LoadInst>();
            auto loadInst2=val2->dynCast<ir::LoadInst>();
            auto ptr1=loadInst1->ptr();
            auto ptr2=loadInst2->ptr();
            if(ptr1!=ptr2){
                return dPossiblySame | dCrossIterPossiblySame;
            }
            else{
                return dPossiblySame | dCrossIterPossiblySame;//进行副作用分析可以进一步细化
            }
        }

        case iELSE:
        {
            return dPossiblySame | dCrossIterPossiblySame;
        }
            
        default:
            break;
        }
    }
    return dPossiblySame | dCrossIterPossiblySame;
}