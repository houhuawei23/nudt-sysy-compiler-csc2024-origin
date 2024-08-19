#include "pass/pass.hpp"

#include "pass/analysis/dom.hpp"
#include "pass/analysis/CFGAnalysis.hpp"
#include "pass/optimize/mem2reg.hpp"
#include "pass/optimize/DCE.hpp"
#include "pass/optimize/SCP.hpp"
#include "pass/optimize/SCCP.hpp"
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
#include "pass/analysis/indvar.hpp"
#include "pass/optimize/GlobalToLocal.hpp"
#include "pass/optimize/TCO.hpp"
#include "pass/optimize/InstCombine/ArithmeticReduce.hpp"
#include "pass/analysis/CFGPrinter.hpp"
#include "pass/optimize/DSE.hpp"
#include "pass/optimize/DLE.hpp"
#include "pass/optimize/SCEV.hpp"
#include "pass/optimize/loopunroll.hpp"
#include "pass/optimize/loopsplit.hpp"
#include "pass/optimize/loopdivest.hpp"
#include "pass/optimize/DAE.hpp"
#include "pass/optimize/licm.hpp"
#include "pass/analysis/dependenceAnalysis/DependenceAnalysis.hpp"
#include "pass/analysis/MarkParallel.hpp"
#include "pass/optimize/GepSplit.hpp"
#include "pass/optimize/aggressiveG2L.hpp"
#include "pass/optimize/indvarEndvarRepl.hpp"

#include "pass/optimize/Misc/StatelessCache.hpp"

#include "pass/optimize/Loop/LoopBodyExtract.hpp"
#include "pass/optimize/Loop/ParallelBodyExtract.hpp"
#include "pass/optimize/Loop/LoopInterChange.hpp"
#include "pass/optimize/Loop/LoopParallel.hpp"
#include "pass/optimize/Misc/BlockSort.hpp"

#include "support/config.hpp"
#include "support/FileSystem.hpp"
#include <fstream>
#include <iostream>
#include <cassert>

namespace pass {

template <typename PassType, typename Callable>
void runPass(PassType* pass, Callable&& runFunc, const std::string& passName) {
    const auto& config = sysy::Config::getInstance();
    auto start = std::chrono::high_resolution_clock::now();
    runFunc();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    double threshold = 1e-3;

    if (elapsed.count() > threshold and config.logLevel >= sysy::LogLevel::DEBUG) {
        std::cout << passName << " " << pass->name() << " took " << elapsed.count() << " seconds.\n";
        // auto fileName = utils::preName(config.infile) + "_after_" + passName + ".ll";
    }
}

void PassManager::run(ModulePass* mp) {
    runPass(mp, [&]() { mp->run(irModule, tAIM); }, "ModulePass");
}

void PassManager::run(FunctionPass* fp) {
    runPass(
      fp,
      [&]() {
          for (auto func : irModule->funcs()) {
              if (func->isOnlyDeclare()) continue;
              fp->run(func, tAIM);
          }
      },
      "FunctionPass");
}

void PassManager::run(BasicBlockPass* bp) {
    runPass(
      bp,
      [&]() {
          for (auto func : irModule->funcs()) {
              for (auto bb : func->blocks()) {
                  bp->run(bb, tAIM);
              }
          }
      },
      "BasicBlockPass");
}
static LoopParallel loopParallelPass;
static StatelessCache cachePass;
static irCheck irCheckPass;
static CFGAnalysisHHW cfgAnalysisPass;
static LoopBodyExtract loopBodyExtractPass;
static ParallelBodyExtract parallelBodyExtractPass;
static BlockSort blockSortPass;
static LoopInterChange loopInterChangePass;

void PassManager::runPasses(std::vector<std::string> passes) {
    // if(passes.size() == 0) return;

    const auto& config = sysy::Config::getInstance();

    if (config.logLevel >= sysy::LogLevel::DEBUG) {
        std::cerr << "Running passes: ";
        for (auto pass_name : passes) {
            std::cerr << pass_name << " ";
        }
        std::cerr << std::endl;
        auto fileName = utils::preName(config.infile) + "_before_passes.ll";
        config.dumpModule(irModule, fileName);
    }

    run(new pass::CFGAnalysisHHW());

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
            run(new pass::SCCP());
        } else if (pass_name.compare("simplifycfg") == 0) {
            run(new pass::simplifyCFG());
        } else if (pass_name.compare("loopanalysis") == 0) {
            run(new pass::loopAnalysis());
            run(new pass::loopInfoCheck());
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
        } else if (pass_name.compare("indvar") == 0) {
            run(new pass::indVarAnalysis());
            run(new pass::indVarInfoCheck());
        } else if (pass_name.compare("g2l") == 0) {
            run(new pass::global2local());
        } else if (pass_name.compare("tco") == 0) {
            run(new pass::tailCallOpt());
        } else if (pass_name.compare("cfgprint") == 0) {
            run(new pass::CFGPrinter());
        } else if (pass_name.compare("licm") == 0) {  // 必须配合gcm使用才能unroll gcm gvn licm unroll
            run(new pass::LICM());
        } else if (pass_name.compare("dse") == 0) {
            run(new pass::simpleDSE());
        } else if (pass_name.compare("dle") == 0) {
            run(new pass::simpleDLE());
        } else if (pass_name.compare("scev") == 0) {
            run(new pass::SCEV());
        } else if (pass_name.compare("unroll") == 0) {
            run(new pass::loopUnroll());
        } else if (pass_name.compare("loopsplit") == 0) {
            run(new pass::loopsplit());
        } else if (pass_name.compare("loopdivest") == 0) {
            run(new pass::loopdivest());
        } else if (pass_name.compare("dae") == 0) {
            run(new pass::DAE());
        } else if (pass_name == "parallel") {
            run(&loopParallelPass);
        } else if (pass_name == "cache") {
            run(&cachePass);
            // run(&cfgAnalysisPass); run in cache pass
        } else if (pass_name == "GepSplit") {
            run(new pass::GepSplit());
        } else if (pass_name == "LoopInterChange") {
            run(&loopInterChangePass);
        } else if (pass_name == "LoopBodyExtract") {
            run(&loopBodyExtractPass);
        } else if (pass_name == "ParallelBodyExtract") {
            run(&parallelBodyExtractPass);
        } else if (pass_name == "blocksort") {
            run(&blockSortPass);
        } else if (pass_name == "da") {
            run(new pass::dependenceAnalysis());
        } else if (pass_name == "ag2l") {
            run(new pass::aggressiveG2L());
        } else if (pass_name == "idvrepl") {
            run(new pass::idvEdvRepl());
        } else if (pass_name == "markpara") {
            run(new pass::markParallel());
        } else {
            std::cerr << "Invalid pass name: " << pass_name << std::endl;
            assert(false && "Invalid pass name");
        }
        if (config.logLevel >= sysy::LogLevel::DEBUG) {
            auto fileName = utils::preName(config.infile) + "_after_" + pass_name + ".ll";
            config.dumpModule(irModule, fileName);
        }
        run(&irCheckPass);
    }
    run(new pass::CFGAnalysisHHW());

    if (config.logLevel >= sysy::LogLevel::DEBUG) {
        auto fileName = utils::preName(config.infile) + "_after_passes.ll";
        config.dumpModule(irModule, fileName);
    }

    irModule->rename();
}

}  // namespace pass