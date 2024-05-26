#include "ir/ir.hpp"
#include "pass/pass.hpp"
#include <vector>
#include <set>

namespace pass{
    class callGraphBuild : public ModulePass{
        public:
            void run(ir::Module* ctx)override;
            std::string name()override;
        private:
            std::vector<ir::Function*>funcStack;
            std::set<ir::Function*>funcSet;
            std::map<ir::Function*,bool>vis;
            void dfsFuncCallGraph(ir::Function*func);

    };
}