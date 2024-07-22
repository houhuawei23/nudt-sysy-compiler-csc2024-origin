#include "pass/optimize/optimize.hpp"
#include "pass/optimize/mem2reg.hpp"
#include <set>
#include <cassert>
#include <map>
#include <vector>
#include <queue>
#include <algorithm>
namespace pass {
// 删除Allocas中指定下标的变量
void Mem2Reg::RemoveFromAllocasList(unsigned& AllocaIdx) {
    Allocas[AllocaIdx] = Allocas.back();
    Allocas.pop_back();
    AllocaIdx--;
}

// 分析一个变量，遍历其使用，是store就把store的BB插入到定义块集合，是load就把load的BB插入到定义块集合
void Mem2Reg::allocaAnalysis(ir::AllocaInst* alloca) {
    // clear
    // OnlyStore = nullptr;
    // OnlyUsedInOneBlock = true;
    // OnlyBlock = nullptr;

    for (auto use : alloca->uses()) {
        ir::Instruction* User = dyn_cast<ir::Instruction>(use->user());
        if (auto store = dynamic_cast<ir::StoreInst*>(User)) {
            DefsBlock[alloca].insert(store->block());
            // DefsBlockvector[alloca].push_back(store->block());
            // OnlyStore = store;
        }

        else if (auto load = dynamic_cast<ir::StoreInst*>(User)) {
            UsesBlock[alloca].insert(load->block());
            // UsesBlockvector[alloca].push_back(load->block());
        }

        // if (OnlyUsedInOneBlock)
        // {
        //     if (!OnlyBlock)
        //         OnlyBlock = User->block();
        //     else if (OnlyBlock != User->block())
        //         OnlyUsedInOneBlock = false;
        // }
    }
}
// 判断变量能否被mem2reg，主要是判断类型是否符合
bool Mem2Reg::is_promoted(ir::AllocaInst* alloca) {
    auto allocapt = dyn_cast<ir::PointerType>(alloca->type())->baseType();
    for (const auto& use : alloca->uses()) {
        if (auto load = dynamic_cast<ir::LoadInst*>(use->user())) {
            if (load->type() != allocapt) {
                return false;
            }
        } else if (auto store = dynamic_cast<ir::StoreInst*>(use->user())) {
            // 这里type的比较要比较其指针的basetype而不是本身
            if (store->value() == alloca ||
                store->value()->type() != allocapt) {
                return false;
            }
        } else {
            return false;
        }
    }
    return true;
}
// // 计算BB中的一条store指令是第几条指令(序号从0开始)
// int Mem2Reg::getStoreinstindexinBB(ir::BasicBlock *BB, ir::StoreInst *I)
// {
//     int index = 0;
//     for (auto &inst : BB->insts())
//     {
//         if (dyn_cast<ir::StoreInst>(inst) == I)
//             return index;
//         index++;
//     }
//     return -1;
// }
// // 计算BB中的一条load指令是第几条指令(序号从0开始)
// int Mem2Reg::getLoadeinstindexinBB(ir::BasicBlock *BB, ir::LoadInst *I)
// {
//     int index = 0;
//     for (auto &inst : BB->insts())
//     {
//         if (dyn_cast<ir::LoadInst>(inst) == I)
//             return index;
//         index++;
//     }
//     return -1;
// }
// // 计算一个BB中变量AI有多少个store
// int Mem2Reg::getStoreNuminBB(ir::BasicBlock *BB, ir::AllocaInst *AI)
// {
//     int num = 0;
//     for (auto inst : BB->insts())
//     {
//         if (auto store = dyn_cast<ir::StoreInst>(inst))
//         {
//             if (store->ptr() == AI)
//                 num++;
//         }
//     }
//     return num;
// }
// // 找出一个BB中变量AI的最后一个store
// ir::StoreInst *Mem2Reg::getLastStoreinBB(ir::BasicBlock *BB, ir::AllocaInst
// *AI)
// {
//     ir::StoreInst *LastStoreinst;
//     for (auto iter = BB->insts().rbegin(); iter != BB->insts().rend();
//     iter++)
//     {
//         auto inst = *iter;
//         if (auto store = dyn_cast<ir::StoreInst>(inst))
//         {
//             if (store->ptr() == AI)
//             {
//                 LastStoreinst = store;
//                 break;
//             }
//         }
//     }
//     return LastStoreinst;
// }
// 处理onlystore的变量，思路是先统计该onlystore的BB中有多少个该变量的store，然后自底向上处理
// 对于每个store，为了删除其对应的load，首先遍历该变量alloca的使用，如果是store就不管；
// 是load分两种情况：一是这条store和load在同一块，二是不同块
// 同一块需要判断store是不是在load之前，之前才能替换load的所有使用然后删除load，不然就跳过
// 不在同一块需要判断store所在块是否支配load所在块，支配才能替换load的所有使用然后删除load，不然就跳过
// 如果最后这个变量变成useempty的，说明rewrite成功
// bool Mem2Reg::rewriteSingleStoreAlloca(ir::AllocaInst *alloca)
// {
//     bool not_globalstore;
//     int StoreIndex;
//     ir::BasicBlock *storeBB = OnlyStore->block();
//     ir::LoadInst *LD;
//     UsesBlock[alloca].clear();
//     // UsesBlockvector[alloca].clear();
//     not_globalstore = not ir::isa<ir::GlobalVariable>(OnlyStore->ptr());
//     StoreIndex = -1;
//     for (auto institer = alloca->uses().begin(); institer !=
//     alloca->uses().end();)
//     {
//         auto inst = (*institer)->user();
//         institer++;
//         if (dyn_cast<ir::StoreInst>(inst))
//         {
//             continue;
//         }

//         ir::LoadInst *load = dyn_cast<ir::LoadInst>(inst);
//         if (not_globalstore)
//         {
//             if (load->block() == storeBB)
//             {
//                 if (StoreIndex == -1)
//                 {
//                     StoreIndex = getStoreinstindexinBB(storeBB, OnlyStore);
//                 }
//                 if (StoreIndex > getLoadeinstindexinBB(storeBB, load))
//                 {
//                     UsesBlock[alloca].insert(storeBB);
//                     // UsesBlockvector[alloca].push_back(storeBB);
//                     continue;
//                 }
//             }
//             else if (!storeBB->dominate(load->block())) //
//             如果storeBB並未支配load則不能进行替换
//             {
//                 UsesBlock[alloca].insert(load->block());
//                 // UsesBlockvector[alloca].push_back(load->block());
//                 continue;
//             }

//         }
//         ir::Value *ReplVal = OnlyStore->value();
//         load->replaceAllUseWith(ReplVal);
//         load->block()->delete_inst(load);
//     }

//     if (!UsesBlock[alloca].empty())
//         return false;
//     storeBB->delete_inst(OnlyStore);
//     alloca->block()->delete_inst(alloca);
//     return true;
// }

// bool Mem2Reg::pormoteSingleBlockAlloca(ir::AllocaInst *alloca)
// {
//     // std::vector<std::pair<int,ir::StoreInst*>> StoresByIndex;
//     return false;
// }
void Mem2Reg::insertphi() {
    // 遍历所有alloca，对于每个alloca，在所有定义块的支配边界中插入phi指令
    std::set<ir::BasicBlock*> Phiset;
    std::vector<ir::BasicBlock*> W;
    ir::PhiInst* phi;
    ir::BasicBlock* x;
    for (ir::AllocaInst* alloca : Allocas) {
        Phiset.clear();
        W.clear();
        phi = nullptr;
        for (ir::BasicBlock* BB : DefsBlock[alloca]) {
            W.push_back(BB);
        }
        while (!W.empty()) {
            x = W.back();
            W.pop_back();
            for (ir::BasicBlock* Y : domctx->domfrontier(x)) {//x->domFrontier
                if (Phiset.find(Y) == Phiset.end()) {
                    auto allocabaseType =
                        dyn_cast<ir::PointerType>(alloca->type())->baseType();
                    phi = new ir::PhiInst(Y, allocabaseType);
                    allphi.push_back(phi);
                    Y->emplace_first_inst(phi);
                    Phiset.insert(Y);
                    PhiMap[Y].insert({phi, alloca});
                    if (find(DefsBlock[alloca].begin(), DefsBlock[alloca].end(),
                             Y) == DefsBlock[alloca].end())
                        W.push_back(Y);
                }
            }
        }
    }
}

void Mem2Reg::rename(ir::Function* F) {
    // rename:填充phi指令内容
    std::vector<ir::Instruction*> instRemovelist;
    std::stack<
        std::pair<ir::BasicBlock*, std::map<ir::AllocaInst*, ir::Value*>>>
        Worklist;
    std::set<ir::BasicBlock*> VisitedSet;
    ir::BasicBlock *SuccBB, *BB;
    std::map<ir::AllocaInst*, ir::Value*> Incommings;
    ir::Instruction* Inst;
    Worklist.push({F->entry(), {}});  // 用栈来做dfs
    for (ir::AllocaInst* alloca : Allocas) {
        Worklist.top().second[alloca] = ir::Constant::gen_undefine();
    }
    while (!Worklist.empty()) {
        BB = Worklist.top().first;
        Incommings = Worklist.top().second;
        Worklist.pop();
        if (VisitedSet.find(BB) != VisitedSet.end())
            continue;
        else
            VisitedSet.insert(BB);

        for (auto inst : BB->insts()) {
            if (ir::AllocaInst* AI = dyn_cast<ir::AllocaInst>(inst)) {
                if (find(Allocas.begin(), Allocas.end(), AI) ==
                    Allocas
                        .end())  // 如果不是可提升的alloca就不管，否则把这条alloca放入待删除列表
                    continue;
                instRemovelist.push_back(inst);
            }

            else if (ir::LoadInst* LD = dyn_cast<ir::LoadInst>(inst)) {
                ir::AllocaInst* AI = dyn_cast<ir::AllocaInst>(LD->operand(0));
                if (!AI)
                    continue;
                if (find(Allocas.begin(), Allocas.end(), AI) != Allocas.end()) {
                    if (Incommings.find(AI) ==
                        Incommings.end())  // 如果这条alloca没有到达定义
                    {
                        Incommings[AI] = ir::Constant::gen_undefine();
                    }
                    LD->replaceAllUseWith(Incommings[AI]);
                    instRemovelist.push_back(inst);
                }
            }

            else if (ir::StoreInst* ST = dyn_cast<ir::StoreInst>(inst)) {
                ir::AllocaInst* AI = dyn_cast<ir::AllocaInst>(ST->ptr());
                if (!AI)
                    continue;
                if (find(Allocas.begin(), Allocas.end(), AI) == Allocas.end())
                    continue;
                Incommings[AI] = ST->value();
                instRemovelist.push_back(inst);
            }

            else if (ir::PhiInst* PHI = dyn_cast<ir::PhiInst>(inst)) {
                if (PhiMap[BB].find(PHI) == PhiMap[BB].end())
                    continue;
                Incommings[PhiMap[BB][PHI]] = PHI;
            }
        }

        for (auto& sBB : BB->next_blocks()) {
            SuccBB = dyn_cast<ir::BasicBlock>(sBB);
            Worklist.push({SuccBB, Incommings});

            for (auto inst : SuccBB->insts()) {
                if (ir::PhiInst* PHI = dyn_cast<ir::PhiInst>(inst)) {
                    if (PhiMap[SuccBB].find(PHI) == PhiMap[SuccBB].end())
                        continue;
                    if (Incommings[PhiMap[SuccBB][PHI]] != nullptr) {
                        PHI->addIncoming(Incommings[PhiMap[SuccBB][PHI]], BB);
                    }
                }
            }
        }
    }
    while (!instRemovelist.empty()) {
        Inst = instRemovelist.back();
        Inst->block()->delete_inst(Inst);
        instRemovelist.pop_back();
    }
    for (auto& item : PhiMap)
        for (auto& pa : item.second) {
            if (pa.first->uses().empty())
                pa.first->block()->delete_inst(pa.first);
        }

    for (ir::PhiInst *phiinst : allphi){
        simplifyphi(phiinst);
    }
}

void Mem2Reg::simplifyphi(ir::PhiInst *phi){
    ir::Value* preval = nullptr;
    ir::BasicBlock* bb = phi->block();
    for (size_t i = 0; i < phi->getsize(); i++){
        if (preval == nullptr)
            preval = phi->getValue(i);
        else{
            if (preval != phi->getValue(i))
                return;
        }
            
    }
    phi->replaceAllUseWith(preval);
    bb->delete_inst(phi);
    return;
}
void Mem2Reg::promotememToreg(ir::Function* F) {
    // 预处理Allocas中不能被mem2reg的变量(没有use的变量和onlystore的变量，onlystore的变量只有唯一的到达定义，不能形成phi指令)
    for (unsigned int AllocaNum = 0; AllocaNum != Allocas.size(); AllocaNum++) {
        ir::AllocaInst* ai = Allocas[AllocaNum];
        if (ai->uses().empty())  // 没有use的变量
        {
            ai->block()->delete_inst(ai);
            RemoveFromAllocasList(AllocaNum);
            continue;
        }
        allocaAnalysis(ai);  // 计算这个变量的使用块集合和定义块集合
        // if (DefsBlockvector[ai].size() == 1) // 如果onlystore
        // {
        //     if (rewriteSingleStoreAlloca(ai)) //
        //     如果返回true，说明这个变量更新后没有use,直接从allocas删除
        //     {
        //         RemoveFromAllocasList(AllocaNum);
        //         continue;
        //     }
        // }
        // if(OnlyUsedInOneBlock and pormoteSingleBlockAlloca(ai))
        // {
        //     RemoveFromAllocasList(AllocaNum);
        //     continue;
        // }
    }
    // 插入phi指令
    insertphi();
    // 填充phi指令
    // F->print(std::cout);
    rename(F);
    // F->print(std::cout);

}

// 主函数
// 首先遍历函数F的第一个块取出所有alloca，如果alloca的basetype是float32或i32或i1，再判断这个alloca是否可做mem2reg，可以就加入Allocas；
// 如果Allocas是empty的就直接break，否则进入promotememToreg函数对F做mem2reg；
bool Mem2Reg::promotemem2reg(ir::Function* F) {
    bool changed = false;
    while (true) {
        Allocas.clear();
        ir::BasicBlock* bb = F->entry();
        for (auto& inst : bb->insts()) {
            if (auto* ai = dyn_cast<ir::AllocaInst>(inst)) {
                // 这里不是ai->type()->is_xx(),
                // 而应该是其指针原来的类型->is_xx()
                // auto aitype = ai->type();
                // if (aitype and aitype->isPointer()) {
                //     auto pttype = dyn_cast<ir::PointerType>(aitype);
                //     auto aibasetype = pttype->baseType();
                //     if (aibasetype->isFloat32() or aibasetype->isInt32() or
                //         aibasetype->isBool() or aibasetype->isArray()) {
                //         if (is_promoted(ai))
                //             Allocas.push_back(ai);
                //     }
                // }
                if (ai->isScalar()) {
                    
                    // if (is_promoted(ai)) {
                        // ai->print(std::cerr); std::cerr << "\n";
                        Allocas.push_back(ai);
                    // }
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

void Mem2Reg::run(ir::Function* F, topAnalysisInfoManager* tp) {
    if (not F->entry())
        return;
    domctx=tp->getDomTree(F);
    domctx->refresh();
    Allocas.clear();
    DefsBlock.clear();
    UsesBlock.clear();
    PhiMap.clear();
    ValueMap.clear();
    allphi.clear();
    promotemem2reg(F);
}
}  // namespace pass