#include "pass/optimize/optimize.hpp"
#include "pass/optimize/mem2reg.hpp"
#include <set>
#include <cassert>
#include <map>
#include <vector>
#include <queue>
#include <algorithm>
namespace pass
{
    void Mem2Reg::RemoveFromAllocasList(unsigned &AllocaIdx)
    {
        Allocas[AllocaIdx] = Allocas.back();
        Allocas.pop_back();
        AllocaIdx--;
    }

    void Mem2Reg::allocaAnalysis(ir::AllocaInst *alloca)
    {
        for (const auto &use : alloca->uses())
        {
            if (auto store = dynamic_cast<ir::StoreInst *>(use->user()))
            {
                DefsBlock[alloca].push_back(store->parent());
                OnlyStore = store;
                
            }

            else if (auto load = dynamic_cast<ir::StoreInst *>(use->user()))
                UsesBlock[alloca].push_back(load->parent());
        }
    }

    bool Mem2Reg::is_promoted(ir::AllocaInst *alloca)
    {
        auto allocapt=dyn_cast<ir::PointerType>(alloca->type())->base_type();
        for (const auto &use : alloca->uses())
        {
            if (auto load = dynamic_cast<ir::LoadInst *>(use->user()))
            {
                if (load->type() != allocapt)
                {
                    return false;
                }
            }
            else if (auto store = dynamic_cast<ir::StoreInst *>(use->user()))
            {
                // 这里type的比较要比较其指针的basetype而不是本身, 估计这整个程序里面都有这样的问题
                if (store->value() == alloca || store->value()->type() != allocapt)
                {
                    return false;
                }
            }
            else
            {
                return false;
            }
        }
        return true;
    }

    int Mem2Reg::getStoreinstindexinBB(ir::BasicBlock *BB, ir::StoreInst *I)
    {
        int index = 0;
        for (auto &inst : BB->insts())
        {
            if (dyn_cast<ir::StoreInst>(inst) == I)
                return index;
            index++;
        }
        return -1;
    }

    int Mem2Reg::getLoadeinstindexinBB(ir::BasicBlock *BB, ir::LoadInst *I)
    {
        int index = 0;
        for (auto &inst : BB->insts())
        {
            if (dyn_cast<ir::LoadInst>(inst) == I)
                return index;
            index++;
        }
        return -1;
    }

    bool Mem2Reg::rewriteSingleStoreAlloca(ir::AllocaInst *alloca)
    {
        bool not_globalstore = not ir::isa<ir::GlobalVariable>(OnlyStore->value());
        int StoreIndex = -1;
        ir::BasicBlock *storeBB = OnlyStore->parent();
        UsesBlock[alloca].clear();
        // int StoreCnt=0;
        for (auto institer=alloca->uses().begin();institer!=alloca->uses().end();)
        {
            auto inst=(*institer)->user();
            institer++;
            if (dyn_cast<ir::StoreInst>(inst)){
                // StoreCnt++;
                continue;
            }
                
            ir::LoadInst *load = dyn_cast<ir::LoadInst>(inst);
            if (not_globalstore)
            {
                if (load->parent() == storeBB)
                {
                    if (StoreIndex == -1)
                    {
                        StoreIndex = getStoreinstindexinBB(storeBB, OnlyStore);
                    }
                    if (StoreIndex > getLoadeinstindexinBB(storeBB, load))
                    {
                        UsesBlock[alloca].push_back(storeBB);
                        continue;
                    }
                }
                else if (not storeBB->dominate(load->parent()))// 如果storeBB支配了load不能进行替换
                {
                    UsesBlock[alloca].push_back(load->parent());
                    continue;
                }
            }
            // 
            ir::Value *ReplVal = OnlyStore->value();
            load->replace_all_use_with(ReplVal);
            load->parent()->delete_inst(load);
            
        }
        if (!UsesBlock[alloca].empty())
            return false;
        OnlyStore->parent()->delete_inst(OnlyStore);
        // if(StoreCnt==1)
        alloca->parent()->delete_inst(alloca);
        return true;
    }

    void Mem2Reg::promotememToreg(ir::Function *F)
    {
        for (unsigned int AllocaNum = 0; AllocaNum != Allocas.size(); AllocaNum++)
        {
            ir::AllocaInst *ai = Allocas[AllocaNum];
            if (ai->uses().empty())
            {
                ai->parent()->delete_inst(ai);
                RemoveFromAllocasList(AllocaNum);
                DeadallocaNum++;
                continue;
            }
            allocaAnalysis(ai);
            if (DefsBlock[ai].size() == 1)
            {
                if (rewriteSingleStoreAlloca(ai))
                {
                    RemoveFromAllocasList(AllocaNum);
                    continue;
                }
            }
        }

        std::set<ir::BasicBlock *> Phiset;
        std::vector<ir::BasicBlock *> W;
        ir::PhiInst *phi;
        ir::BasicBlock *x;
        /// 插入phi节点
        for (ir::AllocaInst *alloca : Allocas)
        {
            Phiset.clear();
            W.clear();
            phi = nullptr;
            for (ir::BasicBlock *BB : DefsBlock[alloca])
            {
                W.push_back(BB);
            }
            while (!W.empty())
            {
                x = W.back();
                W.pop_back();
                for (ir::BasicBlock *Y : x->domFrontier)
                {
                    if (Phiset.find(Y) == Phiset.end())
                    {
                        auto allocabaseType=dyn_cast<ir::PointerType>(alloca->type())->base_type();
                        phi = new ir::PhiInst(Y, allocabaseType);
                        Y->emplace_first_inst(phi);
                        Phiset.insert(Y);
                        PhiMap[Y].insert({phi, alloca});
                        if (find(W.begin(), W.end(), Y) == W.end())
                            W.push_back(Y);
                    }
                }
            }
        }
        // rename:填充phi指令内容
        std::vector<ir::Instruction *> instRemovelist;
        std::vector<std::pair<ir::BasicBlock *, std::map<ir::AllocaInst *, ir::Value *>>> Worklist;
        std::set<ir::BasicBlock *> VisitedSet;
        ir::BasicBlock *SuccBB, *BB;
        std::map<ir::AllocaInst *, ir::Value *> Incommings;
        ir::Instruction* Inst;

        Worklist.push_back({F->entry(), {}});
        for (ir::AllocaInst *alloca : Allocas)
        {
            // TODO Worklist[0].second[alloca] = ir::UndefValue::get(alloca->getAllocatedType());
            Worklist[0].second[alloca] = ir::Constant::gen_undefine();
        }
        while (!Worklist.empty())
        {
            
            BB = Worklist.back().first;
            Incommings = Worklist.back().second;
            Worklist.pop_back();

            if (VisitedSet.find(BB) != VisitedSet.end())
                continue;
            else
                VisitedSet.insert(BB);
            
            for (auto inst : BB->insts())
            {

                if (ir::AllocaInst *AI = dyn_cast<ir::AllocaInst>(inst))
                {
                    if (find(Allocas.begin(), Allocas.end(), AI) == Allocas.end())//如果不是可提升的alloca就不管，否则把这条alloca放入待删除列表
                        continue;
                    instRemovelist.push_back(inst);
                }

                else if (ir::LoadInst *LD = dyn_cast<ir::LoadInst>(inst))
                {
                    ir::AllocaInst *AI = dyn_cast<ir::AllocaInst>(LD->operand(0));
                    if (!AI)
                        continue;
                    if (find(Allocas.begin(), Allocas.end(), AI) != Allocas.end())
                    {
                        if (Incommings.find(AI) == Incommings.end())//如果这条alloca没有到达定义
                        {
                            // TODO 
                            Incommings[AI] = ir::Constant::gen_undefine();
                        }
                        LD->replace_all_use_with(Incommings[AI]);
                        instRemovelist.push_back(inst);
                    }
                }

                else if (ir::StoreInst *ST = dyn_cast<ir::StoreInst>(inst))
                {
                    ir::AllocaInst *AI = dyn_cast<ir::AllocaInst>(ST->ptr());
                    if (!AI)
                        continue;
                    if (find(Allocas.begin(), Allocas.end(), AI) == Allocas.end())
                        continue;
                    Incommings[AI] = ST->value();
                    instRemovelist.push_back(inst);
                }

                else if (ir::PhiInst *PHI = dyn_cast<ir::PhiInst>(inst))
                {
                    if (PhiMap[BB].find(PHI) == PhiMap[BB].end())
                        continue;
                    Incommings[PhiMap[BB][PHI]] = PHI;
                }
            }
            
            for (auto &sBB : BB->next_blocks())
            {
                SuccBB = dyn_cast<ir::BasicBlock>(sBB);
                Worklist.push_back({SuccBB, Incommings});
                for (auto inst : SuccBB->insts())
                {
                    if (ir::PhiInst *PHI = dyn_cast<ir::PhiInst>(inst))
                    {
                        if (PhiMap[SuccBB].find(PHI) == PhiMap[SuccBB].end())
                            continue;
                        if (Incommings[PhiMap[SuccBB][PHI]]){
                            PHI->addIncoming(Incommings[PhiMap[SuccBB][PHI]], BB);
                            // F->print(std::cout);
                        }
                            
                    }
                }
            }
        }
        while (!instRemovelist.empty())
        {
            Inst = instRemovelist.back();
            Inst->parent()->delete_inst(Inst);
            instRemovelist.pop_back();
        }
        for (auto &item : PhiMap)
            for (auto &pa : item.second)
            {
                if (pa.first->uses().empty())
                    pa.first->parent()->delete_inst(pa.first);
            }
    }
    //主函数 首先遍历函数F的第一个块取出所有alloca，如果alloca的basetype是float32或i32或i1，再判断这个alloca是否可做mem2reg，可以就加入Allocas；
    //如果Allocas是empty的就直接break，否则进入promotememToreg函数对F做mem2reg；
    bool Mem2Reg::promotemem2reg(ir::Function *F)
    {
        bool changed = false;
        while (true)
        {
            Allocas.clear();
            ir::BasicBlock *bb = F->entry();
            for (auto &inst : bb->insts())
            {
                if (auto *ai = dyn_cast<ir::AllocaInst>(inst))
                {
                    //这里不是ai->type()->is_xx(), 而应该是其指针原来的类型->is_xx()
                    auto aitype=ai->type();
                    if (aitype and aitype->is_pointer())
                    {
                        auto pttype=dyn_cast<ir::PointerType>(aitype);
                        auto aibasetype=pttype->base_type();
                        if(aibasetype->is_float32() or aibasetype->is_i32() or aibasetype->is_i1())
                        {
                            if (is_promoted(ai))
                                Allocas.push_back(ai);
                        }
                        
                    }
                }
            }
            
            if (Allocas.empty())
                break;
            promotememToreg(F);
            changed = true;
        }
        return changed;
    }

    void Mem2Reg::run(ir::Function *F)
    {
        if(not F->entry())return;
        promotemem2reg(F);
    }
} // namespace pass