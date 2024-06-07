#include <set>
#include <cassert>
#include <map>
#include <vector>
#include "ir/ir.hpp"
#include "pass/pass.hpp"

namespace pass
{
    class GCM : public FunctionPass
    {
    private:
        std::set<ir::Instruction *> insts_visited;
        std::map<ir::Instruction *,ir::BasicBlock *> Earlymap;
    public:
        void run(ir::Function *func) override;
        std::string name() override
        {
            return "GCM";
        }

        void scheduleEarly(ir::Instruction *instruction);
        void scheduleLate(ir::Instruction *instruction);
        ir::BasicBlock *LCA(ir::BasicBlock *lhs, ir::BasicBlock *rhs);
        bool ispinned(ir::Instruction *instruction);
    };
}