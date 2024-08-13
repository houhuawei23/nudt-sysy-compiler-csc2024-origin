#include <set>
#include <cassert>
#include <map>
#include <vector>
#include <queue>
#include "ir/ir.hpp"
#include "pass/pass.hpp"

using namespace ir;
namespace pass {

struct LoopBodyFuncInfo {
  CallInst* callInst;
  IndVar* indVar;
  BasicBlock* preHeader;
  BasicBlock* header;
  BasicBlock* body;
  BasicBlock* latch;

  void print(std::ostream& os) const {
    os << "LoopBodyFuncInfo: " << std::endl;
    std::cout << "callInst: ";
    callInst->print(os);
    os << std::endl;
    std::cout << "indVar: ";
    indVar->print(os);
    std::cout << std::endl;
    std::cout << "header: ";
    header->dumpAsOpernd(os);
    os << std::endl;
    std::cout << "body: ";
    body->dumpAsOpernd(os);
    os << std::endl;
    std::cout << "latch: ";
    latch->dumpAsOpernd(os);
    os << std::endl;
    std::cout << "preHeader: ";
    preHeader->dumpAsOpernd(os);
    os << std::endl;
  }
};

class LoopBodyExtract : public FunctionPass {
public:
  void run(Function* func, TopAnalysisInfoManager* tp) override;
  std::string name() const override { return "LoopBodyExtract"; }

private:
  bool runImpl(Function* func, TopAnalysisInfoManager* tp);
};

bool extractLoopBody(Function* func,
                     Loop& loop,
                     IndVar* indVar,
                     TopAnalysisInfoManager* tp,
                     LoopBodyFuncInfo& loopBodyInfo);

}  // namespace pass