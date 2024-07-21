#include "pass/pass.hpp"

#include "pass/analysis/dom.hpp"
#include "pass/analysis/CFGAnalysis.hpp"
#include "pass/optimize/mem2reg.hpp"
#include "pass/optimize/DCE.hpp"
#include "pass/optimize/SCP.hpp"
#include "pass/optimize/mySCCP.hpp"
#include "pass/optimize/simplifyCFG.hpp"
#include "pass/analysis/loop.hpp"
#include "pass/optimize/GCM.hpp"
#include "pass/optimize/GVN.hpp"
#include "pass/analysis/pdom.hpp"
#include "pass/optimize/inline.hpp"
#include "pass/optimize/reg2mem.hpp"
#include "pass/optimize/ADCE.hpp"
#include "pass/optimize/loopsimplify.hpp"
#include "pass/analysis/irtest.hpp"
#include "pass/analysis/ControlFlowGraph.hpp"
#include "pass/analysis/LCSSA.hpp"

#include "pass/optimize/InstCombine/ArithmeticReduce.hpp"
namespace pass {
void PassManager::runPasses(std::vector<std::string> passes) {
  run(new pass::CFGAnalysisHHW());
  if (not passes.empty()) {
    for (auto pass_name : passes) {
      if (pass_name.compare("dom") == 0) {
        run(new pass::domInfoPass());
        // run(new pass::domInfoCheck());
      } else if (pass_name.compare("mem2reg") == 0) {
        run(new pass::Mem2Reg());
      } else if (pass_name.compare("pdom") == 0) {
        run(new pass::postDomInfoPass());
        // run(new pass::postDomInfoCheck());
      } else if (pass_name.compare("dce") == 0) {
        run(new pass::DCE());
      } else if (pass_name.compare("scp") == 0) {
        run(new pass::SCP());
      } else if (pass_name.compare("sccp") == 0) {
        run(new pass::mySCCP());
      } else if (pass_name.compare("simplifycfg") == 0) {
        run(new pass::simplifyCFG());
      } else if (pass_name.compare("loopanalysis") == 0) {
        run(new pass::loopAnalysis());
        // run(new pass::loopInfoCheck());
      } else if (pass_name.compare("gcm") == 0) {
        run(new pass::GCM());
      } else if (pass_name.compare("gvn") == 0) {
        run(new pass::GVN());
      } else if (pass_name.compare("reg2mem") == 0) {
        run(new pass::Reg2Mem());
      } else if (pass_name.compare("inline") == 0) {
        run(new pass::Inline());
      } else if (pass_name.compare("adce") == 0) {
        run(new pass::ADCE());
      } else if (pass_name.compare("loopsimplify") == 0) {
        run(new pass::loopsimplify());
      } else if (pass_name.compare("instcombine") == 0) {
        // recommend: -p mem2reg -p instcombine -p dce
        run(new pass::ArithmeticReduce());
      } else if (pass_name.compare("test") == 0) {
        run(new pass::irCheck());
      } else if (pass_name.compare("LCSSA") == 0){
        run(new pass::LCSSA());
      }
      else {
        assert(false && "Invalid pass name");
      }
    }
  }
}

}  // namespace pass