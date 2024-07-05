#include "pass/optimize/optimize.hpp"
#include "pass/optimize/GVN.hpp"
#include "ir/value.hpp"
#include <set>
#include <cassert>
#include <map>
#include <vector>
#include <queue>
#include <algorithm>

namespace pass
{
    void GVN::dfs(ir::BasicBlock *bb)
    {
        assert(bb != nullptr && "nullptr in GVN");
        visited.insert(bb);
        for (ir::BasicBlock *succ : bb->next_blocks())
        {
            if (visited.find(succ) == visited.end())
            {
                dfs(succ);
            }
        }
        RPOblocks.push_back(bb);
    }

    void GVN::RPO(ir::Function *F)
    {
        RPOblocks.clear();
        visited.clear();
        ir::BasicBlock *root = F->entry();
        // assert(root != nullptr && "Function without entry block");
        dfs(root);
        reverse(RPOblocks.begin(), RPOblocks.end());
    }

    ir::Value *GVN::getValueNumber(ir::Instruction *inst)
    {
        if (auto binary = dynamic_cast<ir::BinaryInst *>(inst))
            return getValueNumber(binary);
        else if (auto unary = dynamic_cast<ir::UnaryInst *>(inst))
            return getValueNumber(unary);
        else if (auto getelementptr = dynamic_cast<ir::GetElementPtrInst *>(inst))
            return getValueNumber(getelementptr);
        // else if (auto phi = dynamic_cast<ir::PhiInst *>(inst))
        //     return getValueNumber(phi);
        // else if (auto call = dynamic_cast<ir::CallInst *>(inst))
        //     return getValueNumber(call);
        // else if (auto alloca = dynamic_cast<ir::AllocaInst *>(inst))
        //     return getValueNumber(alloca);
        // else if (auto icmp = dynamic_cast<ir::ICmpInst *>(inst))
        //     return getValueNumber(icmp);
        // else if (auto fcmp = dynamic_cast<ir::FCmpInst *>(inst))
        //     return getValueNumber(fcmp);
        else
            return nullptr;
    }

    ir::Value *GVN::getValueNumber(ir::BinaryInst *inst)
    {
        auto lhs = checkHashtable(inst->lValue());
        auto rhs = checkHashtable(inst->rValue());
        for (auto [Key, Value] : _Hashtable)
        {
            if (auto binary = dynamic_cast<ir::BinaryInst *>(Key))
            {
                auto binlhs = checkHashtable(binary->lValue());
                auto binrhs = checkHashtable(binary->rValue());
                if (binary->valueId() == inst->valueId() && ((lhs == binlhs && rhs == binrhs) || (binary->isCommutative() && lhs == binrhs && rhs == binlhs)))
                {
                    return Value;
                }
            }
        }
        return static_cast<ir::Value*>(inst);
    }

    ir::Value *GVN::getValueNumber(ir::UnaryInst *inst)
    {
        auto val = checkHashtable(inst->value());
        for (auto [Key, Value] : _Hashtable)
        {
            if (auto unary = dynamic_cast<ir::UnaryInst *>(Key))
            {
                auto unval = checkHashtable(unary->value());

                if (unary->valueId() == inst->valueId() && unval == val) //????
                {
                    return Value;
                }
            }
        }
        return static_cast<ir::Value*>(inst);
    }

    ir::Value *GVN::getValueNumber(ir::GetElementPtrInst *inst)
    {
        auto arval = checkHashtable(inst->value());
        auto aridx = checkHashtable(inst->index());
        for (auto [Key, Value] : _Hashtable)
        {
            if (auto getelementptr = dynamic_cast<ir::GetElementPtrInst *>(Key))
            {
                auto getval = checkHashtable(getelementptr->value());
                auto getidx = checkHashtable(getelementptr->index());

                if (arval == getval && aridx == getidx)
                {
                    return Value;
                }
            }
        }
        return static_cast<ir::Value*>(inst);
    }
    // ir::Value *GVN::getValueNumber(ir::PhiInst *inst)
    // {
    //     for (auto [Key, Value] : _Hashtable)
    //     {
    //         if (auto phi = dynamic_cast<ir::PhiInst *>(Key))
    //         {
    //             if (phi->parent() == inst->parent())
    //             {
    //                 bool same = (phi->getsize() == inst->getsize());
    //                 for (size_t i = 0; i < phi->getsize(); i++)
    //                 {
    //                     auto v1 = phi->getValue(i);
    //                     auto v2 = inst->getValue(i);
    //                     if (v1 != v2)
    //                     {
    //                         same = false;
    //                         break;
    //                     }
    //                 }
    //                 if (same)
    //                 {
    //                     return Value;
    //                 }
    //             }
    //         }
    //     }
    //     return static_cast<ir::Value*>(inst);
    // }

    // ir::Value *GVN::getValueNumber(ir::CallInst *inst)
    // {
    //     return static_cast<ir::Value*>(inst);
    // }

    // ir::Value *GVN::getValueNumber(ir::AllocaInst *inst)
    // {
    //     return static_cast<ir::Value*>(inst);
    // }

    // ir::Value *GVN::getValueNumber(ir::ICmpInst *inst)
    // {
    //     auto lhs = checkHashtable(inst->lhs());
    //     auto rhs = checkHashtable(inst->rhs());
    //     for (auto [Key, Value] : _Hashtable)
    //     {
    //         if (auto icmp = dynamic_cast<ir::ICmpInst *>(Key))
    //         {
    //             auto icmplhs = checkHashtable(icmp->lhs());
    //             auto icmprhs = checkHashtable(icmp->rhs());
    //             if (icmp->type() == inst->type() && (lhs == icmplhs && rhs == icmprhs) || (icmp->isReverse(inst) && lhs == icmprhs && rhs == icmplhs))
    //             {
    //                 return Value;
    //             }
    //         }
    //     }
    //     return static_cast<ir::Value*>(inst);
    // }

    // ir::Value *GVN::getValueNumber(ir::FCmpInst *inst)
    // {
    //     auto lhs = checkHashtable(inst->lhs());
    //     auto rhs = checkHashtable(inst->rhs());
    //     for (auto [Key, Value] : _Hashtable)
    //     {
    //         if (auto fcmp = dynamic_cast<ir::FCmpInst *>(Key))
    //         {
    //             auto fcmplhs = checkHashtable(fcmp->lhs());
    //             auto fcmprhs = checkHashtable(fcmp->rhs());
    //             if (fcmp->type() == inst->type() && (lhs == fcmplhs && rhs == fcmprhs) || (fcmp->isReverse(inst) && lhs == fcmprhs && rhs == fcmplhs))
    //             {
    //                 return Value;
    //             }
    //         }
    //     }
    //     return static_cast<ir::Value*>(inst);
    // }

    ir::Value *GVN::checkHashtable(ir::Value *v)
    {
        auto vnum = _Hashtable.find(v);
        if (vnum != _Hashtable.end())
        {
            return vnum->second;
        }

        // if (auto constant = dynamic_cast<ir::Constant *>(v))
        // {
        //     for (auto [Key, Value] : _Hashtable)
        //     {
        //         if (auto constkey = dynamic_cast<ir::Constant *>(Key))
        //         {
        //             if (constkey->isInt32() && (constkey->i32() == constant->i32()))
        //             {
        //                 return Value;
        //             }
        //             else if (constkey->isFloatPoint() && (constkey->f32() == constant->f32()))
        //             {
        //                 return Value;
        //             }
        //             else if (constkey->isBool() && (constkey->i1() == constant->i1()))
        //             {
        //                 return Value;
        //             }
        //         }
        //     }
        // }

        if (auto inst = dynamic_cast<ir::Instruction *>(v))
        {
            if (auto value = getValueNumber(inst))
            {
                _Hashtable[v] = value;
                return value;
            }
        }
        _Hashtable[v] = v;
        return v;
    }
    
    void GVN::visitinst(ir::Instruction *inst)
    {
        ir::BasicBlock *bb = inst->block();
        // if (auto store = dyn_cast<ir::StoreInst>(inst))
        // {
        //     if (!store->value()->type()->isPointer())
        //         _Hashtable[store] = store;
        //     return;
        // }
        // else if (auto phi = dyn_cast<ir::PhiInst>((inst)))
        // {
        //     bool same = true;
        //     auto first = checkHashtable(phi->getValue(0));
        //     for (size_t i = 0; i < phi->getsize(); i++)
        //     {
        //         if (checkHashtable(phi->getValue(i)) != first)
        //         {
        //             same = false;
        //             break;
        //         }
        //     }
        //     if (same)
        //     {
        //         inst->replaceAllUseWith(first);
        //         NeedRemove.insert(inst);
        //     }
                
        //     return;
        // }
        for (auto use : inst->uses())
        {
            if (auto br = dyn_cast<ir::BranchInst>(use->user()))
            {
                return ;
            }
        }

        auto value = checkHashtable(inst);
        if (inst != value)
        {
            // if (auto constvalue = dyn_cast<ir::Constant>(value))
            // {    
            //     inst->replaceAllUseWith(value);
            //     NeedRemove.insert(inst);
            // }
            // else 
            if (auto instvalue = dyn_cast<ir::Instruction>(value))
            {
                ir::BasicBlock * vbb = instvalue->block();
                if (vbb->dominate(bb))
                {
                    inst->replaceAllUseWith(instvalue);
                    NeedRemove.insert(inst);
                }
            }
            
        }
        return;
    }

    void GVN::run(ir::Function *F)
    {
       
        if (F->blocks().empty())
            return;
        RPO(F);
        visited.clear();
        // F->print(std::cout);
        for (auto bb : RPOblocks)
        {
            for (auto inst : bb->insts())
            {
                visitinst(inst);
            }
        }
        
        for (auto inst : NeedRemove)
        {
            ir::BasicBlock *BB = inst->block();
            BB->delete_inst(inst);
        }
    }


}