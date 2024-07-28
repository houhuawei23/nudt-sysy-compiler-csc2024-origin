#include "pass/analysis/irtest.hpp"
using std::cerr, std::endl;
namespace pass{
    void irCheck::run(ir::Module* ctx, topAnalysisInfoManager* tp){
        ctx->rename();
        bool isPass=true;
        for(auto func:ctx->funcs()){
            if(func->isOnlyDeclare())continue;
            cerr<<"Testing function \""<<func->name()<<"\" ..."<<endl;
            isPass=isPass and runDefUseTest(func);
            isPass=isPass and runCFGTest(func);
            isPass=isPass and runPhiTest(func);
            isPass=isPass and checkFuncInfo(func);
        }
        if(not isPass)
            assert(false && "didn't pass irCheck!");
    }

    bool irCheck::checkDefUse(ir::Value* val){
        bool isPass=true;
        for(auto puse:val->uses()){
            auto muser=puse->user();
            auto mindex=puse->index();
            auto mvalue=puse->value();
            if(muser->operands()[mindex]->value() != mvalue){
                cerr<<"Value \""<<muser->name()<<"\" use index "<<mindex<<" operand \""<<mvalue->name()<<"\" got an error!"<<endl;
                isPass=false;
            }
        }
        auto user=dyn_cast<ir::User>(val);
        if(user==nullptr)return isPass;
        int realIdx=0;
        for(auto op:user->operands()){
            auto muser=op->user();
            auto mvalue=op->value();
            auto mindex=op->index();
            if(mindex!=realIdx){
                cerr<<"User \""<<muser->name()<<"\" real idx="<<realIdx<<" op got a wrong idx="<<mindex<<endl;
                isPass=false;
            }
            int isfound=0;
            for(auto value_s_use:mvalue->uses()){
                if(op==value_s_use){
                    isfound++;
                }
            }
            if(not isfound){
                cerr<<"User \""<<muser->name()
                    <<"\" operand idx="<<realIdx
                    <<" not found in value \""<<mvalue->name()
                    <<"\" 's uses()."<<endl;
                isPass=false;
            }
            if(isfound>1){
                cerr<<"User \""<<muser->name()
                    <<"\" operand idx="<<realIdx
                    <<" got multiple found in value \""<<mvalue->name()
                    <<"\" 's uses()."<<endl;
                isPass=false;
            }
            realIdx++;
        }
        return isPass;
    }

    bool irCheck::runDefUseTest(ir::Function* func){
        bool isPass=true;
        for(auto bb:func->blocks()){
            bool bbOK=checkDefUse(bb);
            for(auto inst:bb->insts()){
                bbOK=bbOK and checkDefUse(inst);
                if(bb==func->entry() and (inst->valueId()!=ir::vALLOCA or inst->valueId()!=ir::vBR)){
                    isPass=false;
                    cerr<<"Entry block has non-alloca inst!"<<endl;
                }
                if(inst->valueId()==ir::vALLOCA and bb!=func->entry()){
                    isPass=false;
                    cerr<<"AllocaInst occur in BB:\""<<bb->name()<<"\" but it's not entry block!"<<endl;
                }
                if(inst->block()!=bb){
                    isPass=false;
                    cerr<<"Inst in BB:\""<<bb->name()<<"\" and can't match its parent block!"<<endl;
                }
            }
            if(not bbOK){
                isPass=false;
                cerr<<"Error occur in BB:\""<<bb->name()<<"\"!"<<endl;
            }
        }
        return isPass;
    }

    bool irCheck::checkPhi(ir::PhiInst* phi){
        bool isPass=true;
        int lim=phi->getsize();
        auto operandSize=phi->operands().size();
        if(lim!=operandSize/2){
            cerr<<"In phi \""<<phi->name()<<"\", operandsize/2 is uneqaul to phi Incoming size."<<endl;
            isPass=false;
        }
        for(size_t i=0;i<lim;i++){
            if(phi->getvalfromBB(phi->getBlock(i))!=phi->getValue(i)){
                cerr<<"In phi \""<<phi->name()<<"\" incoming has a mismatch!"<<endl;
                isPass=false;
            }
        }
        return isPass;
    }

    bool irCheck::runPhiTest(ir::Function* func){
        bool isPass=true;
        for(auto bb:func->blocks()){
            int lim=bb->phi_insts().size();
            auto instIter=bb->insts().begin();
            auto phiIter=bb->phi_insts().begin();
            while(1){
                if(phiIter==bb->phi_insts().end())break;
                if(*instIter!=*phiIter){
                    cerr<<"In BB\""<<bb->name()<<"\", we got a phiinst list error!"<<endl;
                    isPass=false;
                }
                isPass=isPass and checkPhi(dyn_cast<ir::PhiInst>(*phiIter));
                phiIter++;
                instIter++;
            }
            if(dyn_cast<ir::PhiInst>(*instIter)!=nullptr){
                cerr<<"In BB\""<<bb->name()<<"\", we got a phiinst not in phiinst list!"<<endl;
                isPass=false;
            }
        }
        return isPass;
    }   

    bool irCheck::runCFGTest(ir::Function* func){
        std::unordered_map<ir::BasicBlock*,int>bbPreSize;
        for(auto bb:func->blocks())bbPreSize[bb]=0;
        bool isPass=true;
        //check succ
        for(auto bb:func->blocks()){
            auto succSet=std::set<ir::BasicBlock*>();
            auto terminator=dyn_cast<ir::BranchInst>(bb->terminator());
            if(terminator){
                if(terminator->is_cond()){
                    succSet.insert(terminator->iftrue());
                    succSet.insert(terminator->iffalse());
                }
                else{
                    succSet.insert(terminator->dest());
                }
            }
            if(bb->next_blocks().size()!=succSet.size()){
                cerr<<"Block \""<<bb->name()<<"\" got invalid succBlock size!"<<endl;
                isPass=false;
            }
            for(auto bbnext:bb->next_blocks()){
                if(succSet.count(bbnext)==0){
                    cerr<<"Block \""<<bb->name()<<"\" have a wrong succsecor"<<endl;
                    isPass=false;
                }
                else if(succSet.count(bbnext)==2){
                    cerr<<"Block \""<<bb->name()<<"\" have a multiple succsecor"<<endl;
                    isPass=false;
                }
                else{
                    bbPreSize[bbnext]++;
                }
            }
        }
        for(auto bb:func->blocks()){
            if(bb->pre_blocks().size()!=bbPreSize[bb]){
                cerr<<"Block \""<<bb->name()<<"\" got invalid preBlock size!"<<endl;
                isPass=false;
            }
        }
        return isPass;
    }

    bool irCheck::checkFuncInfo(ir::Function* func){
        bool isPass=true;
        if(func->entry()->pre_blocks().size()!=0){
            isPass=false;
            cerr<<"Entry block got predecessors!"<<endl;
        }
        if(func->exit()->next_blocks().size()!=0){
            isPass=false;
            cerr<<"Exit block got successors!"<<endl;
        }
        return isPass;
    }   
    
}