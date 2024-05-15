#include <set>
#include <cassert>
#include <map>
#include <vector>
#include "ir/ir.hpp"
#include "pass/pass.hpp"

class CongruenceClass//等价关系类
{
    ir::Value *value_expr_;
    std::vector<ir::Value *> members_;
    bool CongruenceClass::operator==(const CongruenceClass &other) const
    {
        auto v1 = value_expr_;
        auto v2 = other.value_expr_;
        if (v1 == nullptr && v2 == nullptr)
            return true;
        else if (v1 == nullptr || v2 == nullptr)
            return false;
        if (members_ == other.members_)
            return true;
        return false;
    }
};

namespace pass
{
    
    class GVN : public FunctionPass
    {
    private:
        std::vector<ir::BasicBlock *> RPOblocks;
        std::set<ir::BasicBlock *> visited;
        std::map<ir::BasicBlock *,std::vector<CongruenceClass *>> partitionMap;

    public:
        void run(ir::Function *func) override;
        std::string name() override
        {
            return "NewGVN";
        }
        void RPO(ir::Function *F);
        void dfs(ir::BasicBlock *bb);
        std::vector<CongruenceClass *> join(std::vector<CongruenceClass *> p1, std::vector<CongruenceClass *> p2);
        CongruenceClass *Intersect(CongruenceClass *c1,CongruenceClass *c2);
        std::vector<CongruenceClass *> transfer(ir::Value *v,std::vector<CongruenceClass *> pin);
        ir::PhiInst *ValuePhi(ir::Value *v,std::vector<CongruenceClass *> p);
        ir::Value *Valueexper(ir::Value *v);//得到表达式的值编号
        void DetectEquivalence();
    };
}