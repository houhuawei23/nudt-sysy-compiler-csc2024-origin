#include "pass/analysis/dom.hpp"
#include<set>
#include<map>

static std::unordered_map<ir::BasicBlock*, int> dfsNum;//DFS order of bb
static std::unordered_map<ir::BasicBlock*, bool> vis;//used in DFS in idomGen
static std::vector<ir::BasicBlock*> vertex;//tell vertex according to number
static std::unordered_map<ir::BasicBlock* ,ir::BasicBlock*> parent;//parent of vertex in DFS tree
using bbset = std::set<ir::BasicBlock*>;
static std::unordered_map<ir::BasicBlock*, bbset>bucket;//those points have sdom u

static void clearAllDomInfo(){
    dfsNum.clear();
    vis.clear();
    vertex.clear();
    parent.clear();
    bucket.clear();
}

namespace pass
{
    //dfs整个CFG,计算了各个节点的dfs的父节点parent和dfs序号dfsNum
    void idomGen::dfsBlocks(ir::BasicBlock* bb){
        static int curNum=0;
        if(not vis[bb]){
            vis[bb]=true;
            dfsNum[bb]=curNum++;
            for(auto nextbb:bb->next_blocks()){
                if(not vis[nextbb]){
                    parent[nextbb]=bb;
                    dfsBlocks(nextbb);
                }
            }
        }
    }
    void idomGen::getDFSnum(ir::Function* func){
        for(auto bb:func->blocks()){
            vis[bb]=false;
            bb->sdom=bb;
        }
        dfsBlocks(func->entry());
    }

    void idomGen::run(ir::Function* func){
        getDFSnum(func);

    }

    std::string idomGen::name(){return "idomGen";}

} // namespace pass