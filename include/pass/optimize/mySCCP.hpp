#include <set>
#include <cassert>
#include <map>
#include <vector>
#include <queue>
#include "ir/ir.hpp"
#include "pass/pass.hpp"

namespace pass{
    class mySCCP:public FunctionPass{
        public:
            void run(ir::Function* func)override;
            std::string name()override;
        private:
            bool getExecutableFlag(ir::BasicBlock* a, ir::BasicBlock* b);
            void addConstFlod(ir::Instruction*inst);
            bool getDeadFlag(ir::BasicBlock* a);
            void branchInstFlod(ir::BranchInst* brInst);
            void deleteDeadBlock(ir::BasicBlock* bb);
    };
}