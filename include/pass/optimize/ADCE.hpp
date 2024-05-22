#include "ir/ir.hpp"
#include "pass/pass.hpp"
#include <set>

namespace pass
{
    class ADCE : public FunctionPass
    {
        public:
            void run(ir::Function* func)override;
            std::string name(){return "ADCE";}
        private:
            bool isAlive(ir::Instruction* inst);
            void addAlive(ir::Instruction*inst);
            // void DCE_delete(ir::Instruction* inst);
    };

} // namespace pass

