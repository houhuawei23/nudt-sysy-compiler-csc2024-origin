#include "pass/optimize/SCEV.hpp"
using namespace pass;

void SCEV::run(ir::Function* func,TopAnalysisInfoManager* tp){
    if(func->isOnlyDeclare())return;
    lpctx=tp->getLoopInfo(func);
    idvctx=tp->getIndVarInfo(func);
    sectx=tp->getSideEffectInfo();
    lpctx->refresh();
    idvctx->setOff();
    idvctx->refresh();
    sectx->setOff();
    sectx->refresh();
    for(auto lp:lpctx->loops()){
        runOnLoop(lp);
    }
}

void SCEV::runOnLoop(ir::Loop* lp){
    if(lp->exits().size()>1)return;//不处理多出口
    auto defaultIdv=idvctx->getIndvar(lp);
    if(defaultIdv==nullptr)return;//必须有基础indvar
    if(lp->header()->pre_blocks().size()>2)return;//必须先循环标准化
    normalizeIndVarIcmpAndBr(lp,defaultIdv);
    

}

void SCEV::normalizeIndVarIcmpAndBr(ir::Loop* lp,ir::indVar* idv){
    //标准化分支跳转，保证在icmp为真的时候继续循环，为假的时候离开循环
    auto crucialBr=lp->header()->terminator()->dynCast<ir::BranchInst>();
    if(not lp->blocks().count(crucialBr->iftrue()) and lp->blocks().count(crucialBr->iffalse())){
        ir::BasicBlock* tmpBB;
        tmpBB=crucialBr->iffalse();
        crucialBr->set_iffalse(crucialBr->iftrue());
        crucialBr->set_iftrue(tmpBB);
    }
    assert(lp->blocks().count(crucialBr->iftrue()) and not lp->blocks().count(crucialBr->iffalse()));
    //标准化icmp，保证phi在Op1，endVar在Op2
    auto icmpInst=idv->cmpInst();
    if(icmpInst->operands()[0]->value()->dynCast<ir::PhiInst>()==idv->phiinst()
        and icmpInst->operands()[1]->value()==idv->endValue())
        return;
    icmpInst->setOperand(0,idv->phiinst());
    icmpInst->setOperand(1,idv->endValue());
    assert(icmpInst->operands()[0]->value()->dynCast<ir::PhiInst>()==idv->phiinst()
        and icmpInst->operands()[1]->value()==idv->endValue());
    return;
}
