#include "mir/utils.hpp"
#include "mir/ScheduleModel.hpp"

#include <vector>
#include <unordered_map>
#include <unordered_set>

namespace mir {

static void topDownScheduling(
    MIRBlock& block,
    std::unordered_map<MIRInst*, uint32_t>& degrees,
    std::unordered_map<MIRInst*, std::unordered_set<MIRInst*>>& antiDeps,
    const std::unordered_map<const MIRInst*,
                             std::unordered_map<uint32_t, uint32_t>>& renameMap,
    std::unordered_map<MIRInst*, int32_t>& rank,
    const CodeGenContext& ctx,
    int32_t waitPenalty) {
    /* debug */
    bool debugSched = true;
    auto dumpIssue = [&](MIRInst* inst) {
        auto& instInfo = ctx.instInfo.get_instinfo(inst);
        instInfo.print(std::cerr << "issue ", *inst, true);
        std::cerr << std::endl;
    };
    auto dumpReady = [&](MIRInst* inst) {
        auto& instInfo = ctx.instInfo.get_instinfo(inst);
        instInfo.print(std::cerr << "ready ", *inst, true);
        std::cerr << std::endl;
    };

    auto& model = ctx.scheduleModel;
    auto& scheInfo = model->getMicroArchInfo();

    MIRInstList scheduledInsts; /* scheduled Insts */

    ScheduleState state{renameMap};
    std::list<MIRInst*> schedulePlane; /* ready to schedule insts */

    for (auto& inst : block.insts()) {
        if (degrees[inst] == 0) {
            if(debugSched) 
                dumpReady(inst);
            schedulePlane.push_back(inst);
        }
    }

    uint32_t cycle = 0;
    /* readyTime: cycle when inst is ready to schedule */
    std::unordered_map<MIRInst*, uint32_t> readyTime;

    

    /* try to schedule all insts in block */
    while (scheduledInsts.size() < block.insts().size()) {
        if(debugSched) {
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
                    rank[inst] + (cycle - readyTime[inst]) * waitPenalty;
                return newRank;
            };
            schedulePlane.sort([&](MIRInst* lhs, MIRInst* rhs) {
                return evalRanl(lhs) > evalRanl(rhs);
            });

            while (failedCnt < schedulePlane.size()) {
                auto& inst = schedulePlane.front();
                schedulePlane.pop_front();
                auto& scheClass = model->getInstScheClass(inst->opcode());

                if (scheClass.schedule(state, *inst,
                                       ctx.instInfo.get_instinfo(inst))) {
                    /** inst success scheduled,
                     * add inst to scheduledInsts and update degrees
                     * if new ready, add new to newReadyInsts */
                    if(debugSched) 
                        dumpIssue(inst);

                    scheduledInsts.push_back(inst);

                    for (auto target : antiDeps[inst]) {
                        degrees[target]--;
                        if (degrees[target] == 0) {
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
    /**/
    std::unordered_map<MIRInst*, std::unordered_set<MIRInst*>> antiDeps;

    /* inst -> (operand_idx -> reg_idx) */
    std::unordered_map<const MIRInst*, std::unordered_map<uint32_t, uint32_t>>
        renameMap;

    /* indegree: number of insts it depends on: inst -> degree */
    std::unordered_map<MIRInst*, uint32_t> degrees;

    /** insts that 'touch'(use/def) the register: reg -> insts
     * lastTouch[i]: {def, use, use, ...}
     */
    std::unordered_map<uint32_t, std::vector<MIRInst*>> lastTouch;

    /* the last inst define a register: reg -> inst */
    std::unordered_map<uint32_t, MIRInst*> lastDef;

    /** u depends on v , v anti-depends on u
     * add u to antiDeps[v] and increment degrees[u]
     */
    auto addDep = [&](MIRInst* u, MIRInst* v) {
        if (u == v)
            return;
        if (antiDeps[v].insert(u).second) {
            ++degrees[u];

            // auto& instInfoU = ctx.instInfo.getInstInfo(*u);
            // auto& instInfoV = ctx.instInfo.getInstInfo(*v);
            // instInfoU.print(std::cerr, *u, true);
            // std::cerr << " -> ";
            // instInfoV.print(std::cerr, *v, true);
            // std::cerr << std::endl;
        }
    };
    /**
     *
     * iy: add a[Def], b[Use], c[Use]
     * iz: add b[Def], x[Use], c[Use]
     * i1: sub a[Def], b[Use], c[Use]
     * i2: add d[Def], a[Use], b[Use]
     * i3: mul b[Def], d[Use], e[Use]
     * - i2 depends on i1 (i1 anti-depends on i2), addDep(i2, i1)
     * - WAR, i1, i2 anti-depends on i3, addDep(i3, i1), addDep(i3, i2)
     * - when i3: lastTouch[b] = {iz, i1, i2}, except iy
     */
    MIRInst* lastSideEffect = nullptr;
    MIRInst* lastInOrder = nullptr;
    for (auto& inst : block.insts()) {
        auto& instInfo = ctx.instInfo.get_instinfo(inst);
        for (uint32_t idx = 0; idx < instInfo.operand_num(); ++idx) {
            auto op = inst->operand(idx);
            auto opflag = instInfo.operand_flag(idx);
            if (op->is_reg()) {
                /** before stack allocate, after sa, sobj is replaced by reg */
                if (isOperandStackObject(op)) {
                    continue;
                }
                const auto reg = op->reg();
                renameMap[inst][idx] = reg;

                if (opflag & OperandFlagUse) {
                    /* RAW: read after write (use after def) */
                    if (auto it = lastDef.find(reg); it != lastDef.end())
                        addDep(inst, it->second);
                    lastTouch[reg].push_back(inst);
                }

                if (opflag & OperandFlagDef) {
                    /** WAR: write after read (def after use)
                     * use anti-depends on def, def depends on use,
                     * addDep(def, use)
                     */
                    for (auto use : lastTouch[reg]) {
                        addDep(inst, use);
                    }
                    lastTouch[reg] = {inst};
                    lastDef[reg] = inst;
                }
            }
        }
        if (lastInOrder) {
            addDep(inst, lastInOrder);
        }

        /** SideEffect Inst */
        /**
         * store r1[Use], imm[Metadata](r2[Use]): SideEffect
         * add r1[Def], r2[USe], r3[Use]
         * call fxx: SideEffect, InOrder, Call, Terminator
         */
        if (requireOneFlag(instInfo.inst_flag(), InstFlagSideEffect)) {
            /** this SideEffect inst depends on the last SideEffect inst */
            if (lastSideEffect) {
                addDep(inst, lastSideEffect);
            }
            lastSideEffect = inst;
            if (requireOneFlag(instInfo.inst_flag(), InstFlagInOrder |
                                                         InstFlagCall |
                                                         InstFlagTerminator)) {
                /** sideeffect and inorder inst,
                 * then the inst dependes on all previous insts */
                for (auto& prev : block.insts()) {
                    if (prev == inst)
                        break;
                    addDep(inst, prev);
                }
                lastInOrder = inst;
            }
        }

        std::unordered_map<MIRInst*, int32_t> rank;
        int32_t instIdx = 0;
        for (auto& inst : block.insts()) {
            rank[inst] = --instIdx;
            int32_t waitPenalty = 2;
            topDownScheduling(block, degrees, antiDeps, renameMap, rank, ctx,
                              waitPenalty);
        }
    }
}

void preRASchedule(MIRFunction& func, const CodeGenContext& ctx) {
    for (auto& block : func.blocks()) {
        preRAScheduleBlock(*block, ctx);
    }
}

}  // namespace mir
