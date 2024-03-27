#pragma once
#include <vector>
#include "ir/ir.hpp"

namespace pass {

//! Pass Template
template <typename PassUnit>
class Pass {
   public:
    // pure virtual function, define the api
    virtual void run(PassUnit* pass_unit) = 0;
    virtual std::string name() = 0;
};

// Instantiate Pass Class for Module, Function and BB
using ModulePass = Pass<ir::Module>;
using FunctionPass = Pass<ir::Function>;
using BasicBlockPass = Pass<ir::BasicBlock>;


//! PassManager Template
template <typename PassUnit>
class PassManager : public Pass<PassUnit> {
   protected:
    std::vector<Pass<PassUnit>*> _passes;

   public:
    PassManager() = default;
    virtual ~PassManager() = default;

    virtual std::string name() override { return "PassManager"; }

    virtual void run(PassUnit* pass_unit) override {
        for (auto& pass : _passes) {
            pass->run(pass_unit);
        }
    }

    void add_pass(Pass<PassUnit>* pass) { _passes.emplace_back(pass); }

    std::vector<Pass<PassUnit>*>& passes() { return _passes; }
};

// Instantiate
using ModulePassManager = PassManager<ir::Module>;
using FunctionPassManager = PassManager<ir::Function>;
using BasicBlockPassManager = PassManager<ir::BasicBlock>;



}  // namespace pass