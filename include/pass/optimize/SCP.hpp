#include <set>
#include <cassert>
#include <map>
#include <vector>
#include "ir/ir.hpp"
#include "pass/pass.hpp"

namespace pass{
    class SCP:public FunctionPass{
        public:
            void run(ir::Function* func)override;
            std::string name()override;
        private:
            void addConstFlod(ir::Instruction* inst);
    };
}