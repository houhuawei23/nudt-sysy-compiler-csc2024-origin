#include <set>
#include <cassert>
#include <map>
#include <vector>
#include "ir/ir.hpp"
#include "pass/pass.hpp"

namespace pass {
class Reg2Mem : public FunctionPass {
   private:
    std::vector<ir::PhiInst*> allphi;
    std::vector<int> parent;
    std::vector<int> rank;
    std::map<ir::PhiInst*, ir::AllocaInst*> phiweb;
    std::vector<ir::AllocaInst*> allocasToinsert;
    std::map<ir::PhiInst*, ir::LoadInst*> philoadmap;
   public:
    void run(ir::Function* func) override;
    std::string name() override { return "reg2mem"; }
    void clear(){
        allphi.clear();
        parent.clear();
        rank.clear();
        phiweb.clear();
        allocasToinsert.clear();
        philoadmap.clear();
    };
    void getallphi(ir::Function* func);
    // 并查集算法
    int getindex(ir::PhiInst* phiinst) {
        auto it = std::find(allphi.begin(), allphi.end(), phiinst);
        if (it != allphi.end()) {
            size_t index = std::distance(allphi.begin(), it);
            return index;
        } else {
            return -1;
        }
    }
    int find(int x) {
        return x == parent[x] ? x : (parent[x] = find(parent[x]));
    }
    bool issame(int x, int y) { return find(x) == find(y); }
    void tounion(int x, int y) {
        int f0 = find(x);
        int f1 = find(y);
        if (rank[f0] > rank[f1]) {
            parent[f1] = f0;
        } else {
            parent[f0] = f1;
            if (rank[f0] == rank[f1]) {
                rank[f1]++;
            }
        }
    }
    void DisjSet() {
        for (int i = 0; i < allphi.size(); i++) {
            parent.push_back(i);
            rank.push_back(1);
        }
        for (ir::PhiInst* phiinst : allphi) {
            for (size_t i = 0; i < phiinst->getsize(); i++) {
                ir::Value* val = phiinst->getval(i);
                if (ir::PhiInst* phival = dyn_cast<ir::PhiInst>(val)) {
                    int id0 = getindex(phiinst);
                    int id1 = getindex(phival);
                    if (issame(id0, id1)){
                        continue;
                    }
                    else{
                        tounion(id0, id1);
                    }
                }
            }
        }
    }
};
}  // namespace pass