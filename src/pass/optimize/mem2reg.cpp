#include "pass/optimize/optimize.hpp"

namespace pass {

class Mem2Reg : public FunctionPass {
   public:
    std::string name() override { return "Mem2Reg"; }
    
    // TODO: Implement Mem2Reg
};
}  // namespace pass