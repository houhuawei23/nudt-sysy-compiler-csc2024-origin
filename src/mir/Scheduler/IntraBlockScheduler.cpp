#include "mir/utils.hpp"
#include "mir/ScheduleModel.hpp"
#include <vector>
#include <unordered_map>
#include <unordered_set>

namespace mir {
static void topDownScheduleBlock(MIRBlock& block,
                                  const CodeGenContext& ctx,
                                  BlockScheduleContext& scheduleCtx) {
  /* debug */
  bool debugSched = false;
  auto dumpIssue = [&](MIRInst* inst) {
    auto& instInfo = ctx.instInfo.getInstInfo(inst);
    instInfo.print(std::cerr << "issue ", *inst, true);
    std::cerr << std::endl;
  };
  auto dumpReady = [&](MIRInst* inst) {
    auto& instInfo = ctx.instInfo.getInstInfo(inst);
    instInfo.print(std::cerr << "ready ", *inst, true);
    std::cerr << std::endl;
  };

  auto& model = ctx.scheduleModel;
  auto& scheInfo = model->getMicroArchInfo();

  MIRInstList scheduledInsts; /* scheduled Insts */

  ScheduleState state{scheduleCtx.renameMap};
  MIRInstList schedulePlane; /* ready to schedule insts */

  for (auto& inst : block.insts()) {
    if (scheduleCtx.degrees[inst] == 0) {
      if (debugSched)
        dumpReady(inst);
      schedulePlane.push_back(inst);
    }
  }
  const uint32_t maxBusyCycles = 200;
  uint32_t busyCycle = 0, cycle = 0;
  /* readyTime: cycle when inst is ready to schedule */
  std::unordered_map<MIRInst*, uint32_t> readyTime;

  /* try to schedule all insts in block */
  while (scheduledInsts.size() < block.insts().size()) {
    if (debugSched) {
      std::cerr << "cycle " << cycle << std::endl;
    }

    std::vector<MIRInst*> newReadyInsts;
    /* Simulate issue in one cycle */
    for (uint32_t idx = 0; idx < scheInfo.issueWidth; idx++) {
      // uint32_t issuedCnt = 0;
      uint32_t failedCnt = 0;
      bool success = false;
      auto evalRanl = [&](MIRInst* inst) {
        int32_t newRank =
          scheduleCtx.rank[inst] + (cycle - readyTime[inst]) * scheduleCtx.waitPenalty;
        return newRank;
      };
      schedulePlane.sort([&](MIRInst* lhs, MIRInst* rhs) { return evalRanl(lhs) > evalRanl(rhs); });

      while (failedCnt < schedulePlane.size()) {
        auto inst = schedulePlane.front();
        schedulePlane.pop_front();
        auto& scheClass = model->getInstScheClass(inst->opcode());

        if (scheClass.schedule(state, *inst, ctx.instInfo.getInstInfo(inst))) {
          /** inst success scheduled,
           * add inst to scheduledInsts and update degrees
           * if new ready, add new to newReadyInsts */
          if (debugSched)
            dumpIssue(inst);

          scheduledInsts.push_back(inst);
          busyCycle = 0;
          for (auto target : scheduleCtx.antiDeps[inst]) {
            scheduleCtx.degrees[target]--;
            if (scheduleCtx.degrees[target] == 0) {
              newReadyInsts.push_back(target);
            }
          }
          success = true;
          break;
        }
        /* failed schedule, readd to schedulePlane */
        failedCnt++;
        schedulePlane.push_back(inst);
      }
      /* if all inst in plane not success scheduled, directly break */
      if (not success)
        break;
    }
    /* Issued finished, cycle++ */
    cycle = state.nextCycle();
    busyCycle++;
    if (busyCycle > maxBusyCycles) {
      std::cerr << "failed to schedule inst: ";
      for (auto& inst : schedulePlane) {
        auto& instInfo = ctx.instInfo.getInstInfo(inst);
        instInfo.print(std::cerr, *inst, true);
        std::cerr << std::endl;
      }
      assert(false && "busy cycle too long");
    }
    for (auto& inst : newReadyInsts) {
      if (debugSched)
        dumpReady(inst);
      readyTime[inst] = cycle;
      schedulePlane.push_back(inst);
    }
  }

  block.insts().swap(scheduledInsts);
}

static void preRAScheduleBlock(MIRBlock& block, const CodeGenContext& ctx) {
  auto scheduleCtx = BlockScheduleContext{};

  scheduleCtx.celloctInfo(block, ctx);

  auto& rank = scheduleCtx.rank;
  int32_t instIdx = 0;
  for (auto& inst : block.insts()) {
    rank[inst] = --instIdx;
  }
  /* schedule block */
  scheduleCtx.waitPenalty = 2;
  topDownScheduleBlock(block, ctx, scheduleCtx);
}

void preRASchedule(MIRFunction& func, const CodeGenContext& ctx) {
  for (auto& block : func.blocks()) {
    preRAScheduleBlock(*block, ctx);
  }
}

static void postRAScheduleBlock(MIRBlock& block, const CodeGenContext& ctx) {
  auto scheduleCtx = BlockScheduleContext{};

  scheduleCtx.celloctInfo(block, ctx);

  auto& rank = scheduleCtx.rank;
  std::unordered_map<MIRInst*, std::unordered_set<MIRInst*>> deps;
  std::unordered_map<MIRInst*, uint32_t> deg;
  for (auto inst : block.insts())
    for (auto prev : scheduleCtx.antiDeps[inst]) {
      deps[prev].insert(inst);
      ++deg[prev];
    }

  std::queue<MIRInst*> q;
  for (auto inst : block.insts())
    if (deg[inst] == 0) {
      rank.emplace(inst, 0);
      q.push(inst);
    }

  while (!q.empty()) {
    auto u = q.front();
    q.pop();
    const auto ru = rank[u];
    for (auto v : deps[u]) {
      auto& rv = rank[v];
      rv = std::max(rv, ru + 1);
      if (--deg[v] == 0) {
        q.push(v);
      }
    }
  }
  /* schedule block */
  scheduleCtx.waitPenalty = 0;
  topDownScheduleBlock(block, ctx, scheduleCtx);
}

void postRASchedule(MIRFunction& func, const CodeGenContext& ctx) {
  // return;
  for (auto& block : func.blocks()) {
    postRAScheduleBlock(*block, ctx);
  }
}
}  // namespace mir