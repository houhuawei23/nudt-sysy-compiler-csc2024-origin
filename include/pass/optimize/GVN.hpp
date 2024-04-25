#include <set>
#include <cassert>
#include <map>
#include <vector>
#include "ir/ir.hpp"
#include "pass/pass.hpp"

namespace pass
{
    class GVN : public FunctionPass
    {
    private:
        std::vector<ir::BasicBlock *> RPOblocks;
        std::set<ir::BasicBlock *> visited;

    public:
        void run(ir::Function *func) override;
        std::string name() override
        {
            return "GVN";
        }
        void postorder(ir::Function *F);
        void dfs(ir::BasicBlock *bb);
    };
}