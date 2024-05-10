#include "pass/optimize/simplifyCFG.hpp"

/*
Performs dead code elimination and basic block merging. Specifically
 - Removes basic blocks with no predecessors.----1
 - Merges a basic block into its predecessor if there is only one and the predecessor only has one
 successor.----2
 - Eliminates PHI nodes for basic blocks with a single predecessor.----3
 - Eliminates a basic block that only contains an unconditional branch.----4
*/


namespace pass{
    std::string simplifyCFG::name(){return "simplifyCFG";}

    void simplifyCFG::run(ir::Function* func){
        if(!func->entry())return;
        // func->print(std::cout);
        bool isWhile=true;
        while(isWhile){
            isWhile=false;
            isWhile|=removeNoPreBlock(func);
            isWhile|=MergeBlock(func);
            isWhile|=removeSingleBrBlock(func);
        }

        
    }

    ir::BasicBlock* simplifyCFG::getSingleDest(ir::BasicBlock* bb){//condition 4
        if(bb->insts().size()!=1)return nullptr;
        auto brInst=dyn_cast<ir::BranchInst>(*(bb->insts().begin()));
        if(brInst and not brInst->is_cond())return brInst->dest();
        return nullptr;
    }

    bool simplifyCFG::noPredBlock(ir::BasicBlock* bb){//condition 1
        return bb->pre_blocks().size()==0 and bb->parent()->entry()!=bb;
    }

    ir::BasicBlock* simplifyCFG::getMergeBlock(ir::BasicBlock* bb){// condition 2
        if(bb->next_blocks().size()==1)
            if((*(bb->next_blocks().begin()))->pre_blocks().size()==1)
                return *(bb->next_blocks().begin());

        return nullptr;
    }

    bool simplifyCFG::MergeBlock(ir::Function *func)
    {
        bool ischanged=false;
        for(auto bb:func->blocks()){
            // func->print(std::cout);
            auto mergeBlock=getMergeBlock(bb);
            while(mergeBlock){
                if(not ischanged)ischanged=true;
                //去掉两个块的联系
                ir::BasicBlock::delete_block_link(bb,mergeBlock);
                //删除最后一条跳转指令
                bb->delete_inst(bb->insts().back());
                //将下一个bb的所有语句复制
                for(auto inst:mergeBlock->insts()){
                    bb->emplace_inst(bb->insts().end(),inst);
                }
                //将下一个bb的所有后继与当前进行连接
                for(auto mergeBBNextIter=mergeBlock->next_blocks().begin();mergeBBNextIter!=mergeBlock->next_blocks().end();){
                    auto mergeBBNext=*mergeBBNextIter;
                    mergeBBNextIter++;
                    ir::BasicBlock::delete_block_link(mergeBlock,mergeBBNext);
                    ir::BasicBlock::block_link(bb,mergeBBNext);
                }
                //将所有的对mergeBB的使用进行替换
                mergeBlock->replace_all_use_with(bb);
                //将merge块删掉
                //因为这些语句没有消失,不能使用一般的delete接口直接删除use
                mergeBlock->insts().clear();
                func->blocks().remove(mergeBlock);
                mergeBlock=getMergeBlock(bb);
            }
        }
        return ischanged;
    }

    bool simplifyCFG::removeNoPreBlock(ir::Function* func){
        bool ischanged=false;
        std::vector<ir::BasicBlock*>worklist;
        for(auto bb:func->blocks()){
            if(noPredBlock(bb))worklist.push_back(bb);
        }
        while(not worklist.empty()){
            auto curBB=worklist.back();
            worklist.pop_back();
            for(auto nextbb:curBB->next_blocks()){
                if(nextbb->pre_blocks().size()==1)
                    worklist.push_back(nextbb);
            }
            func->force_delete_block(curBB);
            if(not ischanged)ischanged=true;
        }
        return ischanged;
    }

    bool simplifyCFG::removeSingleBrBlock(ir::Function* func){
        bool ischanged=false;
        static std::vector<ir::BasicBlock*>singleBrBlocks;
        for(auto bb:func->blocks()){
            if(getSingleDest(bb) and bb!=func->entry())singleBrBlocks.push_back(bb);
        }
        while(not singleBrBlocks.empty()){
            auto curBB=singleBrBlocks.back();
            singleBrBlocks.pop_back();
            auto destBB=getSingleDest(curBB);
            //因为一个块的前驱最复杂的情况也就三四个,没有必要调用十分复杂的接口
            bool isMerge=true;
            bool issamePre=false;
            for(auto preBB1:curBB->pre_blocks()){
                for(auto preBB2:destBB->pre_blocks()){
                    if(preBB1==preBB2){
                        issamePre=true;
                        for(auto inst:destBB->phi_insts()){
                            auto phiInst=dyn_cast<ir::PhiInst>(inst);
                            if(phiInst->getvalfromBB(preBB1)!=phiInst->getvalfromBB(curBB)){
                                isMerge=false;
                                break;
                            }
                        }
                        if(not isMerge)break;
                    }
                }
            }
            if(isMerge){
                ischanged=true;
                if(not issamePre){//如果原来的块和目标块没有共同的pre
                    //修改CFG
                    // func->print(std::cout);
                    for(auto curBBPreBBIter=curBB->pre_blocks().begin();curBBPreBBIter!=curBB->pre_blocks().end();){
                        auto curBBPreBB=*curBBPreBBIter;
                        curBBPreBBIter++;
                        ir::BasicBlock::delete_block_link(curBBPreBB,curBB);
                        ir::BasicBlock::block_link(curBBPreBB,destBB);
                        //添加对应的curBB的前驱到destBB的到达定值
                        for(auto inst:destBB->phi_insts()){
                            auto phiInst=dyn_cast<ir::PhiInst>(inst);
                            phiInst->addIncoming(phiInst->getvalfromBB(curBB),curBBPreBB);
                        }
                    }
                    //将对应phi中的有关destBB的到达定值删除
                    for(auto inst:destBB->phi_insts()){
                        auto phiInst=dyn_cast<ir::PhiInst>(inst);
                        phiInst->delbb(curBB);
                    }
                    ir::BasicBlock::delete_block_link(curBB,destBB);
                    //删除curBB和dest之间的链接
                    //所有跳转到curBB的语句全部跳转到destBB
                    for(auto puse:curBB->uses()){
                        auto inst=dyn_cast<ir::Instruction>(puse->user());
                        auto brInst=dyn_cast<ir::BranchInst>(inst);
                        auto phiInst=dyn_cast<ir::PhiInst>(inst);
                        assert(not phiInst);//保证目前为止的所有使用都是br的,不是phi的,因为我们已经删除了curBB所有下游的phi
                        assert(brInst);
                    }
                    //上面的测试本身是没有必要的,这是为了提前在测试的时候发现错误!
                    //要达到这个效果直接replace_all_use_with就可以了
                    curBB->replace_all_use_with(destBB);
                    func->delete_block(curBB);
                }
                else{//有共同的前驱,而且可以合并的话
                    auto destBBPreBlocks=destBB->pre_blocks();
                    for(auto preBB:curBB->pre_blocks()){//这个循环处理phi
                        bool isDestBBPre=false;
                        for(auto destPreBB:destBBPreBlocks){
                            if(destPreBB==preBB){
                                isDestBBPre=true;
                                break;
                            }
                        }
                        if(isDestBBPre){//是destBB的前驱
                            for(auto inst:destBB->phi_insts()){
                                auto phiinst=dyn_cast<ir::PhiInst>(inst);
                                assert(phiinst->getvalfromBB(destBB)==phiinst->getvalfromBB(curBB));
                            }
                            //这里不用做任何事,这一段代码仅仅作为测试!
                        }
                        else{//不是destBB的前驱
                            for(auto inst:destBB->phi_insts()){
                                auto phiinst=dyn_cast<ir::PhiInst>(inst);
                                phiinst->addIncoming(phiinst->getvalfromBB(curBB),preBB);
                            }
                        }   

                    }//对于所有curBB的前驱,是dest前驱的和不是dest前驱的要分开考虑  
                    //修改CFG
                    for(auto curBBPreBBIter=curBB->pre_blocks().begin();curBBPreBBIter!=curBB->pre_blocks().end();){
                        auto curBBPreBB=*curBBPreBBIter;
                        curBBPreBBIter++;
                        ir::BasicBlock::delete_block_link(curBBPreBB,curBB);
                        bool isConnect=false;
                        for(auto curBBPreBBNextBB:curBBPreBB->next_blocks()){
                            if(curBBPreBBNextBB==destBB){
                                isConnect=true;
                                break;
                            }
                        }
                        //如果是dest的前驱
                        if(isConnect){
                            //需要将其修改为无条件跳转
                            auto lastInst=curBBPreBB->insts().back();
                            auto lastBrInst=dyn_cast<ir::BranchInst>(lastInst);
                            assert(lastBrInst);
                            curBBPreBB->delete_inst(lastInst);
                            curBBPreBB->emplace_inst(curBBPreBB->insts().end(),new ir::BranchInst(destBB,curBBPreBB));
                        }
                        else{
                            ir::BasicBlock::block_link(curBBPreBB,destBB);
                        }
                    }
                    //将phi中的关于curBB的定值删除
                    for(auto inst:destBB->phi_insts()){
                        auto phiinst=dyn_cast<ir::PhiInst>(inst);
                        phiinst->delval(curBB);
                    }
                    curBB->replace_all_use_with(destBB);
                    func->delete_block(curBB);
                }
            }
        }
        return ischanged;
    }
}