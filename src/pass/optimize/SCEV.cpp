#include "pass/optimize/SCEV.hpp"
using namespace pass;

static std::vector<ir::PhiInst*>phiworklist;

void SCEV::run(ir::Function* func,TopAnalysisInfoManager* tp){
    if(func->isOnlyDeclare())return;
    domctx=tp->getDomTree(func);
    lpctx=tp->getLoopInfo(func);
    idvctx=tp->getIndVarInfo(func);
    sectx=tp->getSideEffectInfo();
    
    phiworklist.clear();
    
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
    auto lpHeader=lp->header();
    phiworklist.clear();
    for(auto pinst:lpHeader->phi_insts()){
        if(pinst->uses().empty())continue;
        phiworklist.push_back(pinst->dynCast<ir::PhiInst>());
    }
    while(phiworklist.empty()){
        auto newPhi=phiworklist.back();
        phiworklist.pop_back();
        visitPhi(lp,newPhi);

    }

}

void SCEV::visitPhi(ir::Loop* lp,ir::PhiInst* phiinst){

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
    //-1 不能计算
    assert(idv->isEndVarConst());
    auto beginVar=idv->getBeginI32();
    auto endVar=idv->getEndVarI32();
    auto stepVar=idv->getStepI32();
    auto icmpinst=idv->cmpInst();
    auto iterinst=idv->iterInst();
    if(stepVar==0)return -1;
    //对icmp进行标准化
    normalizeIcmpAndBr(lp,idv);
    switch (icmpinst->valueId())
    {
    case ir::vIEQ:
        if(beginVar==endVar)return 1;
        else return 0;
        break;
    case ir::vINE:
        if(iterinst->valueId()==ir::vADD){
            if((endVar-beginVar)%stepVar!=0)return -1;
            auto cnt=(endVar-beginVar)/stepVar;
            if(cnt<0)return -1;
            return cnt;
        }
        else if(iterinst->valueId()==ir::vSUB){
            if((beginVar-endVar)%stepVar!=0)return -1;
            auto cnt=-(endVar-beginVar)/stepVar;
            if(cnt<0)return -1;
            return cnt;
        }
        else{//MUL
            return -1;//TODO: do not support != with MUL
        }
        break;
    case ir::vISGT:
        if(iterinst->valueId()==ir::vADD){
            if(stepVar>0)return -1;
            if(endVar>=beginVar)return -1;
            auto cnt=(endVar-beginVar)/stepVar;
            if((endVar-beginVar)%stepVar==0)return cnt;
            else return cnt+1;
        }
        else if(iterinst->valueId()==ir::vSUB){
            if(stepVar<0)return -1;
            if(beginVar<=endVar)return -1;
            auto cnt=(beginVar-endVar)/stepVar;
            if((beginVar-endVar)%stepVar==0)return cnt;
            else return cnt+1;
        }
        else if(iterinst->valueId()==ir::vMUL){
            return -1;//TODO: do not support != with MUL
        }
        else{
            assert(false and "invalid operator in IndVar!");
        }
        break;
    case ir::vISGE:
        if(iterinst->valueId()==ir::vADD){
            if(stepVar>0)return -1;
            if(endVar>=beginVar)return -1;
            auto cnt=(endVar-beginVar)/stepVar;
            return cnt+1;
        }
        else if(iterinst->valueId()==ir::vSUB){
            if(stepVar<0)return -1;
            if(beginVar<=endVar)return -1;
            auto cnt=(beginVar-endVar)/stepVar;
            return cnt+1;
        }
        else if(iterinst->valueId()==ir::vMUL){
            return -1;//TODO: do not support != with MUL
        }
        else{
            assert(false and "invalid operator in IndVar!");
        }
        break;
    case ir::vISLT:
        if(iterinst->valueId()==ir::vADD){
            if(stepVar<0)return -1;
            if(endVar<=beginVar)return -1;
            auto cnt=(endVar-beginVar)/stepVar;
            if((endVar-beginVar)%stepVar==0)return cnt;
            else return cnt+1;
        }
        else if(iterinst->valueId()==ir::vSUB){
            if(stepVar>0)return -1;
            if(beginVar<=endVar)return -1;
            auto cnt=(beginVar-endVar)/stepVar;
            if((beginVar-endVar)%stepVar==0)return cnt;
            else return cnt+1;
        }
        else if(iterinst->valueId()==ir::vMUL){
            return -1;//TODO: do not support != with MUL
        }
        else{
            assert(false and "invalid operator in IndVar!");
        }
        break;
    case ir::vISLE:
        if(iterinst->valueId()==ir::vADD){
            if(stepVar<0)return -1;
            if(endVar<=beginVar)return -1;
            auto cnt=(endVar-beginVar)/stepVar;
            return cnt+1;
        }
        else if(iterinst->valueId()==ir::vSUB){
            if(stepVar>0)return -1;
            if(beginVar<=endVar)return -1;
            auto cnt=(beginVar-endVar)/stepVar;
            return cnt+1;
        }
        else if(iterinst->valueId()==ir::vMUL){
            return -1;//TODO: do not support != with MUL
        }
        else{
            assert(false and "invalid operator in IndVar!");
        }
        break;
    default:
        break;
    }
    
    
}

//如果不是常数，就要在必要的时候生成计算迭代次数的指令
//return nullptr if cannot calcuate
ir::Value* SCEV::addCalcIterCntInstructions(ir::Loop* lp,ir::indVar* idv){//-1 for cannot calc
    assert(not idv->isEndVarConst());
    auto beginVar=idv->getBeginI32();
    auto stepVar=idv->getStepI32();
    auto icmpinst=idv->cmpInst();
    auto iterinst=idv->iterInst();
    if(stepVar==0)return nullptr;
    if(lp->exits().size()!=1)return nullptr;
    auto lpExit=*lp->exits().begin();
    auto endVal=idv->endValue();
    auto beginVal=idv->getBegin();
    auto stepVal=idv->getStep();
    //对icmp进行标准化
    normalizeIcmpAndBr(lp,idv);
    //对于不能确定具体cnt的，只生成stepVar==1的情况，否则可以生成所有的情况
    ir::IRBuilder builder;
    builder.set_pos(lpExit,lpExit->insts().begin());
    switch (icmpinst->valueId())
    {
    case ir::vIEQ:
        return nullptr;
        break;
    case ir::vINE:
        return nullptr;
        break;
    case ir::vISGT:
        if(iterinst->valueId()==ir::vADD){
            // if(stepVar>0)return -1;
            // if(endVar>=beginVar)return -1;
            // auto cnt=(endVar-beginVar)/stepVar;
            // if((endVar-beginVar)%stepVar==0)return cnt;
            // else return cnt+1;
            if(stepVar!=-1)return nullptr;
            
            //TODO: makeInst here: sub i32 %beginVal, i32 %endVal;
            return builder.makeBinary(ir::SUB,beginVal,endVal);
        }
        else if(iterinst->valueId()==ir::vSUB){
            // if(stepVar<0)return -1;
            // if(beginVar<=endVar)return -1;
            // auto cnt=(beginVar-endVar)/stepVar;
            // if((beginVar-endVar)%stepVar==0)return cnt;
            // else return cnt+1;
            if(stepVar!=1)return nullptr;

            //TODO: makeInst here: sub i32 %beginVal, i32 %endVal;
            return builder.makeBinary(ir::SUB,beginVal,endVal);

        }
        else if(iterinst->valueId()==ir::vMUL){
            return nullptr;//TODO: do not support != with MUL
        }
        else{
            assert(false and "invalid operator in IndVar!");
        }
        break;
    case ir::vISGE:
        if(iterinst->valueId()==ir::vADD){
            if(stepVar>0)return nullptr;
            // if(endVar>=beginVar)return -1;
            // auto cnt=(endVar-beginVar)/stepVar;
            // return cnt+1;


            //TODO: makeInst here: %newVal = sub i32 %beginVal, i32 %endVal;
            //TODO: makeInst here: %newVal2 = sdiv i32 %newVal, i32 %stepVal;
            //TODO: makeInst here: %newVal2 = add i32 %newVal2, 1
            auto newVal1=builder.makeBinary(ir::SUB,beginVal,endVal);
            ir::Value* newVal2;
            if(stepVar!=1)
                newVal2=builder.makeBinary(ir::DIV,newVal1,stepVal);
            else 
                newVal2=newVal1;
            auto const1=ir::Constant::gen_i32(1);
            return builder.makeBinary(ir::ADD,newVal2,const1);
        }
        else if(iterinst->valueId()==ir::vSUB){
            if(stepVar<0)return nullptr;
            // if(beginVar<=endVar)return -1;
            // auto cnt=(beginVar-endVar)/stepVar;
            // return cnt+1;

            //TODO: makeInst here: %newVal = sub i32 %beginVal, i32 %endVal;
            //TODO: makeInst here: %newVal2 = sdiv i32 %newVal, i32 %stepVal;
            //TODO: makeInst here: %newVal2 = add i32 %newVal2, 1
            ir::Value* newVal2;
            if(stepVar==-1){
                newVal2=builder.makeBinary(ir::SUB,endVal,beginVal);
            }
            else{
                auto newVal1=builder.makeBinary(ir::SUB,beginVal,endVal);
                newVal2=builder.makeBinary(ir::DIV,newVal1,stepVal);
            }
            auto const1=ir::Constant::gen_i32(1);
            return builder.makeBinary(ir::ADD,newVal2,const1);
        }
        else if(iterinst->valueId()==ir::vMUL){
            return nullptr;//TODO: do not support != with MUL
        }
        else{
            assert(false and "invalid operator in IndVar!");
        }
        break;
    case ir::vISLT:
        if(iterinst->valueId()==ir::vADD){
            // if(stepVar<0)return -1;
            // if(endVar<=beginVar)return -1;
            // auto cnt=(endVar-beginVar)/stepVar;
            // if((endVar-beginVar)%stepVar==0)return cnt;
            // else return cnt+1;
            if(stepVar!=1)return nullptr;
            
            //TODO: makeInst here: sub i32 %endVal, i32 %beginVal;
            return builder.makeBinary(ir::SUB,endVal,beginVal);
        }
        else if(iterinst->valueId()==ir::vSUB){
            // if(stepVar>0)return -1;
            // if(beginVar<=endVar)return -1;
            // auto cnt=(beginVar-endVar)/stepVar;
            // if((beginVar-endVar)%stepVar==0)return cnt;
            // else return cnt+1;
            if(stepVar!=-1)return nullptr;
            
            //TODO: makeInst here: sub i32 %endVal, i32 %beginVal;
            return builder.makeBinary(ir::SUB,endVal,beginVal);
        }
        else if(iterinst->valueId()==ir::vMUL){
            return nullptr;//TODO: do not support != with MUL
        }
        else{
            assert(false and "invalid operator in IndVar!");
        }
        break;
    case ir::vISLE:
        if(iterinst->valueId()==ir::vADD){
            if(stepVar<0)return nullptr;
            // if(endVar<=beginVar)return -1;
            // auto cnt=(endVar-beginVar)/stepVar;
            // return cnt+1;

            //TODO: makeInst here: %newVal = sub i32 %endVal, i32 %beginVal;
            //TODO: makeInst here: %newVal2 = sdiv i32 %newVal, i32 %stepVal;
            //TODO: makeInst here: %newVal2 = add i32 %newVal2, 1
            ir::Value* newVal2;
            auto newVal1=builder.makeBinary(ir::SUB,endVal,beginVal);
            if(stepVar==1)
                newVal2=newVal1;
            else
                newVal2=builder.makeBinary(ir::DIV,newVal1,stepVal);
            auto const1=ir::Constant::gen_i32(1);
            return builder.makeBinary(ir::ADD,newVal2,const1);

        }
        else if(iterinst->valueId()==ir::vSUB){
            if(stepVar>0)return nullptr;
            // if(beginVar<=endVar)return -1;
            // auto cnt=(beginVar-endVar)/stepVar;
            // return cnt+1;

            //TODO: makeInst here: %newVal = sub i32 %beginVal, i32 %endVal;
            //TODO: makeInst here: %newVal2 = sdiv i32 %newVal, i32 %stepVal;
            //TODO: makeInst here: %newVal2 = add i32 %newVal2, 1
            ir::Value* newVal2;
            if(stepVar==-1){
                newVal2=builder.makeBinary(ir::SUB,endVal,beginVal);
            }
            else{
                auto newVal1=builder.makeBinary(ir::SUB,beginVal,endVal);
                newVal2=builder.makeBinary(ir::DIV,newVal1,stepVal);
            }
            auto const1=ir::Constant::gen_i32(1);
            return builder.makeBinary(ir::ADD,newVal2,const1);

        }
        else if(iterinst->valueId()==ir::vMUL){
            return nullptr;//TODO: do not support != with MUL
        }
        else{
            assert(false and "invalid operator in IndVar!");
        }
        break;
    default:
        break;
    }
    assert(false and "something error happened in func\" addCalCntInstuctions \"");
    return nullptr;
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