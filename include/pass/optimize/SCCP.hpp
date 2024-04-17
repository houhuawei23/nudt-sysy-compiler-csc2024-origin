#include <set>
#include <cassert>
#include <map>
#include <vector>
#include "ir/ir.hpp"
#include "pass/pass.hpp"

namespace pass{
    class SCCP:public FunctionPass{
        public:
            void run(ir::Function* func)override;
            std::string name()override;
    };
}