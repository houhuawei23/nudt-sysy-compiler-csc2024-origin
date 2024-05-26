#include "ir/ir.hpp"
#include "pass/pass.hpp"
#include <set>
#include <queue>

namespace pass
{
    class ADCE : public FunctionPass
    {
        public:
            void run(ir::Function* func)override;
            std::string name()override{return "ADCE";}
        private:
    };

} // namespace pass

