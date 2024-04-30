#include <set>
#include <cassert>
#include <map>
#include <vector>
#include <queue>
#include "ir/ir.hpp"
#include "pass/pass.hpp"

namespace pass{
    class simplifyCFG:public FunctionPass{
        public:
            void run(ir::Function* func)override;
            std::string name()override;
        private:
            ir::BasicBlock* getSingleDest(ir::BasicBlock* bb);
            bool noPredBlock(ir::BasicBlock* bb);
            ir::BasicBlock* getMergeBlock(ir::BasicBlock* bb);
            bool MergeBlock(ir::Function* func);
            bool removeNoPreBlock(ir::Function* func);
            bool removeSingleBrBlock(ir::Function* func);
    };
}