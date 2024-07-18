#pragma once
#include "mir/mir.hpp"
#include "mir/target.hpp"
#include "mir/CFGAnalysis.hpp"
#include "mir/LiveInterval.hpp"
#include "mir/RegisterAllocator.hpp"
#include "support/StaticReflection.hpp"
#include "target/riscv/riscv.hpp"
#include <vector>
#include <stack>
#include <queue>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>

namespace mir {
struct RegNumComparator final {
    const std::unordered_map<RegNum, double>* weights;
    
    bool operator()(RegNum lhs, RegNum rhs) const {
        return weights->at(lhs) > weights->at(rhs);
    }
};

/*
 * @brief: Interference Graph (干涉图)
 * @note: 
 *      成员变量: 
 *          1. std::unordered_map<RegNum, std::unordered_set<RegNum>> _adj
 *               干涉图 (存储每个节点的邻近节点)
 *          2. std::unordered_map<RegNum, uint32_t> _degree
 *               存储干涉图中虚拟寄存器节点的度数
 *          3. Queue _queue
 *               优先队列, 存储可分配物理寄存器的虚拟寄存器节点
 */
class InterferenceGraph final {
    std::unordered_map<RegNum, std::unordered_set<RegNum>> _adj;
    std::unordered_map<RegNum, uint32_t> _degree;
    using Queue = std::priority_queue<RegNum, std::vector<RegNum>, RegNumComparator>;
    Queue _queue;
public:  /* get function */
    auto& adj(RegNum u) {
        assert(isVirtualReg(u));
        return _adj[u];
    }
public:  /* About Degree Function */
    void create(RegNum u) { if (!_degree.count(u)) _degree[u] = 0U; }
    [[nodiscard]] bool empty() const { return _degree.empty(); }
    [[nodiscard]] size_t size() const { return _degree.size(); }
public:  /* util function */
    /* 功能: 加边 */
    void add_edge(RegNum lhs, RegNum rhs) {
        assert(lhs != rhs);
        assert((isVirtualReg(lhs) || isISAReg(lhs)) && (isVirtualReg(rhs) || isISAReg(rhs)));
        if (_adj[lhs].count(rhs)) return;
        _adj[lhs].insert(rhs); _adj[rhs].insert(lhs);

        /*
         NOTE: 干涉图的节点可以为虚拟寄存器 OR 物理寄存器
         但是我们仅仅只考虑对虚拟寄存器进行相关物理寄存器的指派和分配
         故: 我们仅仅只计算虚拟寄存器节点的度数, 不考虑计算物理寄存器节点的度数
         */
        if (isVirtualReg(lhs)) ++_degree[lhs];
        if (isVirtualReg(rhs)) ++_degree[rhs];
    }
    /* 功能: 为图着色寄存器分配做准备 */
    void prepare_for_assign(const std::unordered_map<RegNum, double>& weights, uint32_t k) {
        _queue = Queue{ RegNumComparator{ &weights } };
        for (auto& [reg, degree] : _degree) {
            if (degree < k) {  /* 度数小于k, 可进行图着色分配 */
                assert(isVirtualReg(reg));
                _queue.push(reg);
            }
        }
    }
    /* 功能: 选择虚拟寄存器来为其分配物理寄存器 */
    RegNum pick_to_assign(uint32_t k) {
        if (_queue.empty()) return invalidReg;
        auto u = _queue.top(); _queue.pop();
        assert(isVirtualReg(u)); assert(adj(u).size() < k);

        if (auto iter = _adj.find(u); iter != _adj.cend()) {
            for (auto v : _adj.at(u)) {
                if (isVirtualReg(v)) {
                    if (_degree[v] == k) _queue.push(v);
                    --_degree[v];
                }
                _adj[v].erase(u);
            }
            _adj.erase(iter);
        }
        _degree.erase(u);
        return u;
    }
    RegNum pick_to_spill(const std::unordered_set<uint32_t>& blockList, const std::unordered_map<RegNum, double>& weights, uint32_t k) const {
        constexpr uint32_t fallbackThreshold = 0;  // 根据blockList的大小选择不同的策略
        RegNum best = invalidReg;
        double minWeight = 1e40;  // 最小权值
        if (blockList.size() >= fallbackThreshold) {  // 策略1: 选择度数最大且权值最小的节点来将其spill到内存中
            uint32_t maxDegree = 0;  // 最大度数
            for (auto& [reg, degree] : _degree) {
                if (degree >= maxDegree && !blockList.count(reg)) {
                    if (maxDegree == degree && weights.at(reg) >= minWeight) continue;
                    maxDegree = degree;
                    minWeight = weights.at(reg);
                    best = reg;
                }
            }
        } else {  // 策略2: 选择权值最小的大于k的节点来将其spill到内存
            for (auto& [reg, degree] : _degree) {
                if (degree >= k && !blockList.count(reg) && weights.at(reg) < minWeight) {
                    best = reg;
                    minWeight = weights.at(reg);
                }
            }
        }
        assert(best != invalidReg && isVirtualReg(best));
        return best;
    }
    auto collect_nodes() const {
        std::vector<RegNum> vregs;
        vregs.reserve(_degree.size());
        for (auto [reg, degree] : _degree) vregs.push_back(reg);
        return vregs;
    }
public:  /* just for debug */
    void dump(std::ostream& out) {
        for (auto& [vreg, degree] : _degree) {
            out << (vreg ^ virtualRegBegin) << "[" << degree << "]: ";
            for (auto adj : _adj[vreg]) {
                if (isVirtualReg(adj)) out << "v";
                else out << "i";
                out << (isVirtualReg(adj) ? adj ^ virtualRegBegin : adj) << " ";
            }
            out << "\n";
        }
    }
};

/*
 * @brief: Graph Coloring Register Allocation (图着色寄存器分配算法)
 * @param: 
 *      1. MIRFunction& mfunc
 *      2. CodeGenContext& ctx
 *      3. IPRAUsageCache& infoIPRA
 *      4. uint32_t allocationClass (决定此次分配是分配整数寄存器还是浮点寄存器)
 *      5. std::unordered_map<uint32_t, uint32_t>& regMap (answer, 存储虚拟寄存器到物理寄存器之间的映射)
 */
static void GraphColoringAllocate(MIRFunction& mfunc, CodeGenContext& ctx, IPRAUsageCache& infoIPRA,  uint32_t allocationClass,  std::unordered_map<uint32_t, uint32_t>& regMap) {
    /* Prepare for Graph Coloring Allocation */
    const auto canonicalizedType = ctx.registerInfo->getCanonicalizedRegisterTypeForClass(allocationClass);
    const auto& list = ctx.registerInfo->get_allocation_list(allocationClass);
    const std::unordered_set<uint32_t> allocableISARegs{ list.cbegin(), list.cend() };
    
    constexpr auto DebugRA = true;
    if (DebugRA) {
        std::cerr << "allocate for class " << allocationClass << std::endl;
    }

    std::unordered_set<uint32_t> blockList;  // 存储spill到内存的寄存器
    bool fixHazard = true;  // --> RISC-V特性

    /* 处理【函数参数的传递】 */
    std::unordered_map<uint32_t, MIROperand> inStackArguments;
    for (auto inst : mfunc.blocks().front()->insts()) {
        if (inst->opcode() == InstLoadRegFromStack) {
            auto dst = inst->operand(0)->reg();
            auto src = inst->operand(1);
            auto& obj = mfunc.stack_objs().at(src);
            if (obj.usage == StackObjectUsage::Argument) {
                inStackArguments.emplace(dst, *src);
            }
        }
    }

    /* 处理【常数】 --> 为什么要进行该处理？？？ */
    /*
     constants: uint32_t -> MIRInst*
        - nullptr: 该虚拟寄存器已被多条指令定义            (删除)
        - inst: 该虚拟寄存器仅仅被InstFlagLoadConstant定义 (存储)
     */
    std::unordered_map<uint32_t, MIRInst*> constants;
    for (auto& block : mfunc.blocks()) {
        for (auto inst : block->insts()) {
            auto& instInfo = ctx.instInfo.get_instinfo(inst);
            
            if (requireFlag(instInfo.inst_flag(), InstFlagLoadConstant)) {
                auto reg = inst->operand(0)->reg();
                if (isVirtualReg(reg)) {
                    if (!constants.count(reg)) constants[reg] = inst;
                    else constants[reg] = nullptr;
                }
            } else {  /* 考虑其他指令 */
                for (uint32_t idx = 0; idx < instInfo.operand_num(); idx++) {
                    auto flag = instInfo.operand_flag(idx);
                    if (!(flag & OperandFlagDef)) continue;
                    auto op = inst->operand(idx);
                    if (isOperandVReg(op)) constants[op->reg()] = nullptr;
                }
            }
        }
    }
    {
        std::vector<uint32_t> eraseKey;
        for (auto [k, v] : constants) {
            if (!v) eraseKey.push_back(k);
        }
        for (auto k : eraseKey) constants.erase(k);
    }

    /* 不断迭代, 开始进行图着色寄存器分配算法 */
    while (true) {
        auto liveInterval = calcLiveIntervals(mfunc, ctx);
        if (DebugRA) mfunc.print(std::cerr, ctx);

        const auto isAllocatableType = [&](OperandType type) {
            return (type <= OperandType::Float32) && (ctx.registerInfo->get_alloca_class(type) == allocationClass);
        };
        const auto isLockedOrUnderRenamedType = [&](OperandType type) { return type <= OperandType::Float32; };
        
        std::unordered_set<RegNum> vregSet;  /* 统计当前所有指令中虚拟寄存器的个数 */
        for (auto& block : mfunc.blocks()) {
            for (auto inst : block->insts()) {
                auto& instInfo = ctx.instInfo.get_instinfo(inst);
                for (uint32_t idx = 0; idx < instInfo.operand_num(); idx++) {
                    auto flag = instInfo.operand_flag(idx);
                    if (!((flag & OperandFlagUse) || (flag & OperandFlagDef))) continue;
                    auto op = inst->operand(idx);
                    if (!(isOperandVReg(op) && isAllocatableType(op->type()))) continue;
                    vregSet.insert(op->reg());
                }
            }
        }

        /* ????????????? */
        std::unordered_map<RegNum, std::set<InstNum>> def_use_time;
        auto colorDefUse = [&](RegNum src, RegNum dst) {
            assert(isVirtualReg(src) && isISAReg(dst));
            if (!fixHazard || !def_use_time.count(src)) return;
            auto& dstInfo = def_use_time[dst];
            auto& srcInfo = def_use_time[src];
            dstInfo.insert(srcInfo.begin(), srcInfo.end());
        };
        std::unordered_map<RegNum, std::unordered_map<RegNum, double>> copy_hint;
        auto updateCopyHint = [&](RegNum dst, RegNum src, double weight) {
            assert(isVirtualReg(dst));
            copy_hint[dst][src] += weight;
        };

        /* Construct Interference Graph */
        InterferenceGraph graph;
        auto cfg = calcCFG(mfunc, ctx);
        auto blockFreq = calcFreq(mfunc, cfg);

        /* ISA Specific Reg */
        for (auto& block : mfunc.blocks()) {
            auto& instructions = block->insts();
            std::unordered_set<uint32_t> underRenamedISAReg;  // 寄存器重命名

            /* 收集待重命名的寄存器 */
            const auto collectUnderRenamedISARegs = [&](MIRInstList::iterator it) {
                while (it != instructions.end()) {
                    auto inst = *it;
                    auto& instInfo = ctx.instInfo.get_instinfo(inst);
                    bool hasReg = false;
                    for (uint32_t idx = 0; idx < instInfo.operand_num(); idx++) {
                        auto op = inst->operand(idx);
                        if (isOperandISAReg(op) && !ctx.registerInfo->is_zero_reg(op->reg()) && 
                            isLockedOrUnderRenamedType(op->type()) && (instInfo.operand_flag(idx) & OperandFlagUse)) {
                            if (isAllocatableType(op->type())) underRenamedISAReg.insert(op->reg());
                            hasReg = true;
                        }
                    }
                    if (hasReg) it++;
                    else break;
                }
            };
            collectUnderRenamedISARegs(instructions.begin());
            
            std::unordered_set<uint32_t> lockedISAReg;  // 被锁住的ISA寄存器
            std::unordered_set<uint32_t> liveVRegs;
            for (auto vreg : liveInterval.block2Info.at(block.get()).ins) {
                assert(isVirtualReg(vreg));
                if (vregSet.count(vreg)) {
                    liveVRegs.insert(vreg);
                }
            }
            
            const auto tripCount = blockFreq.query(block.get());
            for (auto iter = instructions.begin(); iter != instructions.end();) {
                auto next = std::next(iter);
                auto inst = *iter;
                auto& instInfo = ctx.instInfo.get_instinfo(inst);

                if (inst->opcode() == InstCopyFromReg && allocableISARegs.count(inst->operand(1)->reg())) {
                    updateCopyHint(inst->operand(0)->reg(), inst->operand(1)->reg(), tripCount);
                } else if (inst->opcode() == InstCopyToReg && allocableISARegs.count(inst->operand(0)->reg())) {
                    updateCopyHint(inst->operand(1)->reg(), inst->operand(0)->reg(), tripCount);
                } else if (inst->opcode() == InstCopy) {
                    auto u = inst->operand(0)->reg();
                    auto v = inst->operand(1)->reg();
                    if (u != v) {
                        if (isVirtualReg(u)) updateCopyHint(u, v, tripCount);
                        if (isVirtualReg(v)) updateCopyHint(v, u, tripCount);
                    }
                }

                for (uint32_t idx = 0; idx < instInfo.operand_num(); idx++) {
                    if (instInfo.operand_flag(idx) & OperandFlagUse) {
                        auto op = inst->operand(idx);
                        if (!isAllocatableType(op->type())) continue;
                        if (!isOperandVRegORISAReg(op)) continue;

                        def_use_time[op->reg()].insert(liveInterval.inst2Num.at(inst));
                        if (isOperandISAReg(op) && !ctx.registerInfo->is_zero_reg(op->reg())) {
                            underRenamedISAReg.erase(op->reg());
                        } else if (isOperandVReg(op)) {
                            graph.create(op->reg());
                            if (op->reg_flag() & RegisterFlagDead) liveVRegs.erase(op->reg());
                        }
                    }
                }

                /* Call Instruction */
                if (requireFlag(instInfo.inst_flag(), InstFlagCall)) {
                    const IPRAInfo* calleeUsage = nullptr;
                    if (auto symbol = inst->operand(0)->reloc()) {
                        calleeUsage = infoIPRA.query(symbol->name());
                    }
                
                    if (calleeUsage) {
                        for (auto isaReg : *calleeUsage) {
                            if (isAllocatableType(ctx.registerInfo->getCanonicalizedRegisterType(isaReg))) {
                                for (auto vreg : liveVRegs) {
                                    graph.add_edge(vreg, isaReg);
                                }
                            }
                        }
                    } else {
                        for (auto isaReg : ctx.registerInfo->get_allocation_list(allocationClass)) {
                            if (ctx.frameInfo.is_caller_saved(*MIROperand::as_isareg(isaReg, OperandType::Special))) {
                                for (auto vreg : liveVRegs) graph.add_edge(isaReg, vreg);
                            }
                        }
                    }

                    collectUnderRenamedISARegs(next);
                    lockedISAReg.clear();
                }
                for (uint32_t idx = 0; idx < instInfo.operand_num(); idx++) {
                    if (instInfo.operand_flag(idx) & OperandFlagDef) {
                        auto op = inst->operand(idx);
                        if (!isAllocatableType(op->type())) continue;

                        def_use_time[op->reg()].insert(liveInterval.inst2Num.at(inst));
                        if (isOperandISAReg(op) && !ctx.registerInfo->is_zero_reg(op->reg())) {
                            lockedISAReg.insert(op->reg());  // op被锁定
                            for (auto vreg : liveVRegs) graph.add_edge(vreg, op->reg());
                        } else if (isOperandVReg(op)) {
                            liveVRegs.insert(op->reg());
                            graph.create(op->reg());
                            for (auto isaReg : underRenamedISAReg) graph.add_edge(op->reg(), isaReg);
                            for (auto isaReg : lockedISAReg) graph.add_edge(op->reg(), isaReg);
                        }
                    }
                }

                iter = next;
            }
        }

        auto vregs = graph.collect_nodes();
        assert(vregs.size() == vregSet.size());
        if (DebugRA) {
            std::cerr << vregs.size() << std::endl;
        }
        for (size_t i = 0; i < vregs.size(); i++) {
            auto u = vregs[i];
            auto& liveIntervalU = liveInterval.reg2Interval.at(u);

            for (size_t j = i + 1; j < vregs.size(); j++) {
                auto& v = vregs[j];
                auto& liveIntervalV = liveInterval.reg2Interval.at(v);
                if (liveIntervalU.intersectWith(liveIntervalV)) {
                    if (DebugRA) {
                        std::cerr << (u ^ virtualRegBegin) << "<->" << (v ^ virtualRegBegin) << std::endl;
                        liveIntervalU.dump(std::cerr);
                        std::cerr << "\n";
                        liveIntervalV.dump(std::cerr);
                        std::cerr << "\n";
                    }
                    graph.add_edge(u, v);
                }
            }
        }

        if (graph.empty()) return;
        assert(graph.size() == vregs.size());
        if (DebugRA) graph.dump(std::cerr);

        std::vector<std::pair<InstNum, double>> freq;
        for (auto& block : mfunc.blocks()) {
            auto endInst = liveInterval.inst2Num.at(block->insts().back());
            freq.emplace_back(endInst + 2, blockFreq.query(block.get()));
        }
    
        /* Calculate the weights for virtual register */
        std::unordered_map<RegNum, double> weights;  // Weight = \sum (number of use/def) * Freq
        auto getBlockFreq = [&](InstNum inst) {
            auto it = std::lower_bound(freq.begin(), freq.end(), inst, [](const auto& a, const auto& b) { return a.first < b; });
            assert(it != freq.end());
            return it->second;
        };
        for (auto vreg : vregs) {
            auto& liveRange = liveInterval.reg2Interval.at(vreg);
            double weight = 0;
            for (auto& [begin, end] : liveRange.segments) {
                weight += static_cast<double>(end - begin) * getBlockFreq(end);
            }
            if (auto iter = copy_hint.find(vreg); iter != copy_hint.end()) {
                weight += 100.0 * static_cast<double>(iter->second.size());
            }
            if (constants.count(vreg)) weight -= 1.0;
            weights.emplace(vreg, weight);
        }
        for (auto& block : mfunc.blocks()) {
            auto w = blockFreq.query(block.get());
            for (auto inst : block->insts()) {
                auto& instInfo = ctx.instInfo.get_instinfo(inst);
                for (uint32_t idx = 0; idx < instInfo.operand_num(); idx++) {
                    auto flag = instInfo.operand_flag(idx);
                    if (!((flag & OperandFlagUse) || (flag & OperandFlagDef))) continue;
                    auto op = inst->operand(idx);
                    if (!(isOperandVReg(op) && isAllocatableType(op->type()))) continue;
                    weights[op->reg()] += w;
                }
            }
        }

        /* Assign Registers */
        const auto k = static_cast<uint32_t>(list.size());
        std::stack<uint32_t> assignStack;
        bool spillRegister = false;
        auto dynamicGraph = graph;
        dynamicGraph.prepare_for_assign(weights, k);
        while (!dynamicGraph.empty()) {
            auto u = dynamicGraph.pick_to_assign(k);
            if (u == invalidReg) {
                spillRegister = true;
                break;
            }
            if (DebugRA) {
                std::cerr << "push " << (u ^ virtualRegBegin) << std::endl;
            }
            assignStack.push(u);
        }

        if (!spillRegister) {
            const auto calcCopyFreeProposal = [&](RegNum u, std::unordered_set<uint32_t>& exclude) -> std::optional<RegNum> {
                auto iter = copy_hint.find(u);
                if (iter == copy_hint.cend()) return std::nullopt;
                std::unordered_map<RegNum, double> map;
                for (auto [reg, v] : iter->second) {
                    if (isVirtualReg(reg)) {
                        if (auto it = regMap.find(reg); it != regMap.end() && !exclude.count(it->second)) {
                            map[it->second] += v;
                        } else if (!exclude.count(reg)) {
                            map[reg] += v;
                        }
                    }
                }
                if (map.empty()) return std::nullopt;
                double maxWeight = -1e10;
                RegNum best = invalidReg;
                for (auto [reg, v] : map) {
                    if (v > maxWeight) {
                        maxWeight = v;
                        best = reg;
                    }
                }
                if (best == invalidReg) assert(false);
                return best;
            };

            assert(assignStack.size() == vregs.size());
            while (!assignStack.empty()) {
                auto u = assignStack.top(); assignStack.pop();

                std::unordered_set<uint32_t> exclude;
                for (auto v : graph.adj(u)) {
                    if (isVirtualReg(v)) {
                        if (auto iter = regMap.find(v); iter != regMap.end()) {
                            exclude.insert(iter->second);
                        }
                    } else {
                        exclude.insert(v);
                    }
                }

                bool assigned = false;
                if (auto isaReg = calcCopyFreeProposal(u, exclude)) {
                    assert(allocableISARegs.count(*isaReg));
                    regMap[u] = *isaReg;
                    colorDefUse(u, *isaReg);
                    assigned = true;
                } else {
                    if (!fixHazard) {
                        for (auto reg : list) {
                            if (!exclude.count(reg)) {
                                regMap[u] = reg;
                                assigned = true;
                                break;
                            }
                        }
                    } else {
                        RegNum bestReg = invalidReg;
                        double cost = 1e20;
                        auto evalCost = [&](RegNum reg) {
                            assert(isISAReg(reg));
                            constexpr double calleeSavedCost = 5.0;

                            double maxFreq = -1.0;
                            constexpr InstNum infDist = 10;
                            InstNum minDist = infDist;

                            auto& defTimeOfColored = def_use_time[reg];
                            auto evalDist = [&](InstNum curDefTime) {
                                if (defTimeOfColored.empty()) return infDist;
                                auto it = defTimeOfColored.lower_bound(curDefTime);
                                if (it == defTimeOfColored.begin()) return std::min(infDist, *it - curDefTime);
                                if (it == defTimeOfColored.end()) return std::min(infDist, curDefTime - *std::prev(it));
                                return std::min(infDist, std::min(curDefTime - *std::prev(it), *it - curDefTime));
                            };

                            for (auto instNum : def_use_time[u]) {
                                auto f = getBlockFreq(instNum);
                                if (f > maxFreq) {
                                    minDist = evalDist(instNum);
                                    maxFreq = f;
                                } else if (f == maxFreq) {
                                    minDist = std::min(minDist, evalDist(instNum));
                                }
                            }

                            double curCost = -static_cast<double>(minDist);
                            if (ctx.frameInfo.is_callee_saved(*MIROperand::as_isareg(reg, OperandType::Special))) {
                                curCost += calleeSavedCost;
                            }
                            return curCost;
                        };

                        for (auto reg : list) {
                            if (!exclude.count(reg)) {
                                const auto curCost = evalCost(reg);
                                if (curCost < cost) {
                                    cost = curCost;
                                    bestReg = reg;
                                }
                            }
                        }

                        regMap[u] = bestReg;
                        colorDefUse(u, bestReg);
                        assigned = true;
                    }
                }
                if (!assigned) assert(false && "the error occurs in the graph coloring register allocation");

                if (DebugRA) std::cerr << (u ^ virtualRegBegin) << "->" << regMap.at(u) << std::endl;
            }

            return;
        }

        /* Spill Register */
        auto u = graph.pick_to_spill(blockList, weights, k);
        blockList.insert(u);
        if (DebugRA) {
            std::cerr << "spill " << (u ^ virtualRegBegin) << "\n";
            std::cerr << "block list " << blockList.size() << " " << graph.size() << "\n";
        }
        if (!isVirtualReg(u)) {
            assert(false && "the error occurs in the graph coloring register allocation");
        }
        const auto size = getOperandSize(canonicalizedType);
        bool alreadyInStack = inStackArguments.count(u);
        bool rematerializeConstant = constants.count(u);
        MIROperand stackStorage;
        MIRInst* copy_inst = nullptr;
        if (alreadyInStack) {
            stackStorage = inStackArguments.at(u);
        } else if (rematerializeConstant) {
            copy_inst = constants.at(u);
        } else {
            stackStorage = *mfunc.add_stack_obj(ctx.next_id(), size, size, 0, StackObjectUsage::RegSpill);
        }

        std::unordered_set<MIRInst*> newInsts;
        const uint32_t minimizeIntervalThreshold = 8;
        const auto fallback = blockList.size() >= minimizeIntervalThreshold;

        for (auto& block : mfunc.blocks()) {
            auto& instructions = block->insts();

            bool loaded = false;
            for (auto iter = instructions.begin(); iter != instructions.end();) {
                auto next = std::next(iter);
                auto inst = *iter;
                if (newInsts.count(inst)) {
                    iter = next;
                    continue;
                }
                auto& instInfo = ctx.instInfo.get_instinfo(inst);
                bool hasUse = false, hasDef = false;
                for (uint32_t idx = 0; idx < instInfo.operand_num(); idx++) {
                    auto op = inst->operand(idx);
                    if (!isOperandVReg(op)) continue;
                    if (op->reg() != u) continue;

                    auto flag = instInfo.operand_flag(idx);
                    if (flag & OperandFlagUse) hasUse = true;
                    else if (flag & OperandFlagDef) hasDef = true;
                }

                if (hasUse && !loaded) {  // should be placed before locked inst block
                    auto it = iter;
                    while (it != instructions.begin()) {
                        auto& lockedInst = *std::prev(it);
                        auto& lockedInstInfo = ctx.instInfo.get_instinfo(lockedInst);
                        bool hasReg = false;
                        for (uint32_t idx = 0; idx < lockedInstInfo.operand_num(); idx++) {
                            auto op = lockedInst->operand(idx);
                            if (isOperandISAReg(op) && !ctx.registerInfo->is_zero_reg(op->reg()) && 
                                isLockedOrUnderRenamedType(op->type()) && (instInfo.operand_flag(idx) & OperandFlagDef)) {
                                hasReg = true;
                                break;
                            }
                        }

                        if (!hasReg) break;
                        --it;
                    }

                    if (rematerializeConstant) {
                        instructions.insert(it, copy_inst);
                    } else {
                        auto tmp = new MIRInst{ InstLoadRegFromStack };
                        tmp->set_operand(0, MIROperand::as_vreg(u - virtualRegBegin, canonicalizedType));
                        auto tmpStack = new MIROperand(stackStorage);
                        tmp->set_operand(1, tmpStack);
                        instructions.insert(it, tmp);
                    }

                    if (!fallback) loaded = true;
                }
                if (hasDef) {
                    if (alreadyInStack || rematerializeConstant) {
                        instructions.erase(iter);
                        loaded = false;
                    } else {  // should be placed after rename inst block
                        auto it = next;
                        while (it != instructions.end()) {
                            auto renameInst = *it;
                            auto& renameInstInfo = ctx.instInfo.get_instinfo(renameInst);
                            bool hasReg = false;
                            for (uint32_t idx = 0; idx < renameInstInfo.operand_num(); idx++) {
                                auto op = renameInst->operand(idx);
                                if (isOperandISAReg(op) && !ctx.registerInfo->is_zero_reg(op->reg()) && 
                                    isLockedOrUnderRenamedType(op->type()) && (renameInstInfo.operand_flag(idx) & OperandFlagUse)) {
                                    hasReg = true;
                                    break;
                                }
                            }

                            if (!hasReg) break;
                            it++;
                        }
                        auto tmp = new MIRInst{ InstStoreRegToStack };
                        tmp->set_operand(0, MIROperand::as_vreg(u - virtualRegBegin, canonicalizedType));
                        auto tmpStack = new MIROperand(stackStorage);
                        tmp->set_operand(1, tmpStack);
                        auto newInst = instructions.insert(it, tmp);
                        newInsts.insert(*newInst);
                        loaded = false;
                    }
                }
                iter = next;
            }
        }
        cleanupRegFlags(mfunc, ctx);
    }
}

static void GraphColoringAllocate(MIRFunction& mfunc, CodeGenContext& ctx, IPRAUsageCache& infoIPRA) {
    const auto classCount = ctx.registerInfo->get_alloca_class_cnt();
    std::unordered_map<uint32_t, uint32_t> regMap;  // --> 存储[虚拟寄存器]到[物理寄存器]之间的映射
    for (uint32_t idx = 0; idx < classCount; idx++) {
        GraphColoringAllocate(mfunc, ctx, infoIPRA, idx, regMap);
    }

    for (auto& block : mfunc.blocks()) {
        auto& instructions = block->insts();
        for (auto inst : instructions) {
            auto& instInfo = ctx.instInfo.get_instinfo(inst);
            for (uint32_t idx = 0; idx < instInfo.operand_num(); idx++) {
                auto op = inst->operand(idx);
                if (op->type() > OperandType::Float32) continue;
                if (isOperandVReg(op)) {
                    const auto isaReg = regMap.at(op->reg());
                    *op = *MIROperand::as_isareg(isaReg, op->type());
                }
            }
        }
    }
    std::cerr << "regMap's size is " << regMap.size() << std::endl;
}
}