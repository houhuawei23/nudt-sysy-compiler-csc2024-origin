#include "ir/ir.hpp"
#include "pass/pass.hpp"
#include <set>

namespace pass
{
    class DCE : public FunctionPass
    {
        public:
            void run(ir::Function* func)override;
            std::string name(){return "DCE";}
        private:
            bool isAlive(ir::Instruction* inst);
            void addAlive(ir::Instruction*inst);
    };

} // namespace pass

