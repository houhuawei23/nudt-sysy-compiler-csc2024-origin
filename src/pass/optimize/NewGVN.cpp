#include "pass/optimize/optimize.hpp"
#include "pass/optimize/NewGVN.hpp"
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
        visited.insert(bb);
        for(ir::BasicBlock *succ : bb->next_blocks())
        {
            if(visited.find(succ) == visited.end())
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
        dfs(root);
        reverse(RPOblocks.begin(),RPOblocks.end());
    }

//     std::vector<CongruenceClass *> GVN::transfer(ir::Value *v,std::vector<CongruenceClass *> pin)
//     {
        
//     }

//     void GVN::DetectEquivalence()
//     {

//     }
//     void GVN::run(ir::Function *F)
//     {
//         DetectEquivalence();
//     }
}