#include "pass/optimize/SCEV.hpp"
using namespace pass;

void SCEV::run(ir::Function* func,TopAnalysisInfoManager* tp){
    if(func->isOnlyDeclare())return;
    lpctx=tp->getLoopInfo(func);
    idvctx=tp->getIndVarInfo(func);
    sectx=tp->getSideEffectInfo();
    domctx=tp->getDomTree(func);
    lpctx->refresh();
    idvctx->setOff();
    idvctx->refresh();
    sectx->setOff();
    sectx->refresh();
    domctx->refresh();
    for(auto lp:lpctx->loops()){
        if(lp->parent()!=nullptr)continue;//只处理顶层循环，底层循环通过顶层循环向下分析
        runOnLoop(lp);
    }
}

void SCEV::runOnLoop(ir::Loop* lp){
    if(lp->exits().size()>1)return;//不处理多出口
    auto defaultIdv=idvctx->getIndvar(lp);
    // if(defaultIdv==nullptr)return;//必须有基础indvar
    if(not lp->isLoopSimplifyForm())return;
    for(auto subLp:lp->subLoops()){
        runOnLoop(subLp);
    }
    if(defaultIdv==nullptr)return;
    normalizeIcmpAndBr(lp,defaultIdv);
}

//简单的判断一下对应的value是不是循环不变量,其定值如果支配循环头自然就是
bool SCEV::isSimplyLoopInvariant(ir::Loop* lp,ir::Value* val){
    if(auto inst=val->dynCast<ir::Instruction>()){
        auto instBB=inst->block();
        if(domctx->dominate(instBB,lp->header()))return true;
    }
    if(auto conVal=val->dynCast<ir::Constant>())return true;
    return false;
}

//在循环外对这个值(phi)有使用这就说明了这个常数是值得被化简计算的
bool SCEV::isUsedOutsideLoop(ir::Loop* lp,ir::Value* val){
    for(auto puse:val->uses()){
        auto user=puse->user();
        if(auto inst=user->dynCast<ir::Instruction>()){
            if(lp->blocks().count(inst->block())==0)return false;
        }
    }
    return true;
}

//如果endvar是常数，就直接计算出对应的迭代次数
int SCEV::getConstantEndvarIndVarIterCnt(ir::Loop* lp,ir::indVar* idv){
    assert(idv->isEndVarConst());
    auto beginVar=idv->getBeginI32();
    auto endVar=idv->getEndVarI32();
    auto stepVar=idv->getStepI32();
    auto icmpinst=idv->cmpInst();
    //对icmp进行标准化
    normalizeIcmpAndBr(lp,idv);
    
    
}

//如果不是常数，就要在必要的时候生成计算迭代次数的指令
void SCEV::addCalcIterCntInstructions(ir::Loop* lp,ir::indVar* idv){

}

//标准化:把idv放在op1 把endvar放在op2,icmp true就保持循环,false就跳出
void SCEV::normalizeIcmpAndBr(ir::Loop* lp,ir::indVar* idv){
    auto endvar=idv->endValue();
    auto icmpInst=idv->cmpInst()->dynCast<ir::ICmpInst>();
    auto brInst=lp->header()->terminator()->dynCast<ir::BranchInst>();
    assert(icmpInst!=nullptr);
    bool IsIcmpOpNorm=icmpInst->rhs()==endvar;
    bool IsBrDestNorm=lp->blocks().count(brInst->iftrue())>0;
    if(not IsBrDestNorm and not IsIcmpOpNorm){
        std::cerr<<"Lp Br and Icmp both not normalized!"<<std::endl;
        exchangeIcmpOp(icmpInst);
        exchangeBrDest(brInst);
        reverseIcmpOp(icmpInst);
    }
    else if(not IsBrDestNorm){
        std::cerr<<"Lp Br not normalized!"<<std::endl;
        reverseIcmpOp(icmpInst);
        exchangeBrDest(brInst);

    }
    else if(not IsIcmpOpNorm){
        std::cerr<<"Lp Icmp both not normalized!"<<std::endl;
        exchangeIcmpOp(icmpInst);
    }

}

//交换两个Icmp中的Op以使得ind在LHS
void SCEV::exchangeIcmpOp(ir::ICmpInst* icmpInst){
    auto LHS=icmpInst->lhs();
    auto RHS=icmpInst->rhs();
    //改变ValueId
    reverseIcmpOp(icmpInst);
    //交换op
    icmpInst->setOperand(0,RHS);
    icmpInst->setOperand(1,LHS);
}

//翻转这个Icmp的符号使得原意不变
void SCEV::reverseIcmpOp(ir::ICmpInst* icmpInst){
    switch (icmpInst->valueId())
    {
    case ir::vIEQ:
        break;
    case ir::vINE:
        break;
    case ir::vISGE:
        icmpInst->setCmpOp(ir::vISLE);
        break;
    case ir::vISLE:
        icmpInst->setCmpOp(ir::vISGE);
        break;
    case ir::vISLT:
        icmpInst->setCmpOp(ir::vISGT);
        break;
    case ir::vISGT:
        icmpInst->setCmpOp(ir::vISLT);
        break;
    default:
        assert(false and "invalid ICMP Op!");
        break;
    }
}

//交换brcond的跳转两个目标
void SCEV::exchangeBrDest(ir::BranchInst* brInst){
    assert(brInst->is_cond());
    auto trueTarget=brInst->iftrue();
    auto falseTarget=brInst->iffalse();
    brInst->set_iftrue(falseTarget);
    brInst->set_iffalse(trueTarget);
}