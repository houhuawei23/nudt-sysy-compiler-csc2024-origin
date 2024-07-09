#pragma once
#include <array>
#include <list>
#include <variant>
#include <vector>

#include "ir/ir.hpp"

namespace mir {
class MIRRelocable;
class MIRModule;
class MIRFunction;
class MIRBlock;
class MIRInst;
class MIRRegister;
class MIROperand;
struct MIRGlobalObject;
class MIRZeroStorage;
class MIRDataStorage;
struct StackObject;
struct CodeGenContext;

enum CompareOp : uint32_t {
    ICmpEqual,
    ICmpNotEqual,
    ICmpSignedLessThan,
    ICmpSignedLessEqual,
    ICmpSignedGreaterThan,
    ICmpSignedGreaterEqual,
    ICmpUnsignedLessThan,
    ICmpUnsignedLessEqual,
    ICmpUnsignedGreaterThan,
    ICmpUnsignedGreaterEqual,

    FCmpOrderedEqual,
    FCmpOrderedNotEqual,
    FCmpOrderedLessThan,
    FCmpOrderedLessEqual,
    FCmpOrderedGreaterThan,
    FCmpOrderedGreaterEqual,
    FCmpUnorderedEqual,
    FCmpUnorderedNotEqual,
    FCmpUnorderedLessThan,
    FCmpUnorderedLessEqual,
    FCmpUnorderedGreaterThan,
    FCmpUnorderedGreaterEqual
};

/* MIRRelocable */
class MIRRelocable {
    std::string _name;
public:
    MIRRelocable(const std::string& name="") : _name(name) {}
    virtual ~MIRRelocable() = default;
public:  // get function
    auto name() const { return _name; }
public:  // just for debug
    virtual void print(std::ostream& os, CodeGenContext& ctx) = 0;
};

/* utils function */
constexpr uint32_t virtualRegBegin = 0b0101U << 28;
constexpr uint32_t stackObjectBegin = 0b1010U << 28;
constexpr uint32_t invalidReg = 0b1100U << 28;
constexpr bool isISAReg(uint32_t x) { return x < virtualRegBegin; }
constexpr bool isVirtualReg(uint32_t x) { return (x & virtualRegBegin) == virtualRegBegin; }
constexpr bool isStackObject(uint32_t x) { return (x & stackObjectBegin) == stackObjectBegin; }
enum class OperandType : uint32_t {
    Bool,
    Int8,
    Int16,
    Int32,
    Int64,
    Float32,
    Special,
    HighBits,
    LowBits,
    Alignment
};
constexpr bool isIntType(OperandType type) { return type <= OperandType::Int64; }
constexpr bool isFloatType(OperandType type) { return type == OperandType::Float32; }
constexpr uint32_t getOperandSize(const OperandType type) {
    /* NOTE: RISC-V 64位树莓派 */
    switch (type) {
        case OperandType::Int8: return 1;
        case OperandType::Int16: return 2;
        case OperandType::Int32: return 4;
        case OperandType::Int64: return 8;
        case OperandType::Float32: return 4;
        default: assert(false && "invalid operand type");
    }
}

/* MIRRegisterFlag */
enum MIRRegisterFlag : uint32_t {
    RegisterFlagNone = 0,
    RegisterFlagDead = 1 << 1,
};

/* MIRRegister */
class MIRRegister {
    uint32_t _reg;
    MIRRegisterFlag _flag = RegisterFlagNone;
public:
    MIRRegister() = default;
    MIRRegister(uint32_t reg) : _reg(reg) {}
public:  // operator
    bool operator==(const MIRRegister& rhs) const { return _reg == rhs._reg; }
    bool operator!=(const MIRRegister& rhs) const { return _reg != rhs._reg; }
public:  // get function
    uint32_t reg() { return _reg; }
    MIRRegisterFlag flag() { return _flag; }
    MIRRegisterFlag* flag_ptr() { return &_flag; }
public:  // set function
    void set_flag(MIRRegisterFlag flag) { _flag = flag; }
public:
    void print(std::ostream& os);
};

enum MIRGenericInst : uint32_t {
    // Jump
    InstJump,
    InstBranch,
    InstUnreachable,

    // Memory
    InstLoad,
    InstStore,

    // Arithmetic
    InstAdd,
    InstSub,
    InstMul,
    InstUDiv,
    InstURem,

    // Bitwise
    InstAnd,
    InstOr,
    InstXor,
    InstShl,
    InstLShr,  // logic shift right
    InstAShr,  // arth shift right

    // Signed Div/Rem
    InstSDiv,
    InstSRem,

    // MinMax
    InstSMin,
    InstSMax,

    // Unary
    InstNeg,
    InstAbs,

    // Float
    InstFAdd,
    InstFSub,
    InstFMul,
    InstFDiv,
    InstFNeg,
    InstFAbs,
    InstFFma,

    // Cmp
    InstICmp,
    InstFCmp,

    // Conversion
    InstSExt,
    InstZExt,
    InstTrunc,
    InstF2U,
    InstF2S,
    InstU2F,
    InstS2F,
    InstFCast,

    // Misc
    InstCopy,
    InstSelect,
    InstLoadGlobalAddress,
    InstLoadImm,
    InstLoadStackObjectAddr,
    InstCopyFromReg,
    InstCopyToReg, // 45
    InstLoadImmToReg,
    InstLoadRegFromStack,
    InstStoreRegToStack,

    // Return
    InstReturn,

    // ISA Specific
    ISASpecificBegin,
};

/* MIROperand */
class MIROperand {
private:
    std::variant<std::monostate, MIRRegister*, MIRRelocable*, intmax_t, double> _storage{std::monostate{}};
    OperandType _type = OperandType::Special;
public:
    MIROperand() = default;
    template <typename T> MIROperand(T x, OperandType type) : _storage(x), _type(type) {}
public:  // get function
    auto& storage() { return _storage; }
    auto type() { return _type; }
    const intmax_t imm() {
        assert(is_imm());
        return std::get<intmax_t>(_storage);
    }
    double prob() {
        assert(is_prob());
        return std::get<double>(_storage);
    }
    uint32_t reg() const {
        assert(is_reg());
        return std::get<MIRRegister*>(_storage)->reg();
    }
    MIRRelocable* reloc() {
        assert(is_reloc());
        return std::get<MIRRelocable*>(_storage);
    }
    MIRRegisterFlag reg_flag() {
        assert(is_reg() && "the operand is not a register");
        return std::get<MIRRegister*>(_storage)->flag();
    }
public:  // operator
    bool operator==(const MIROperand& rhs) { return _storage == rhs._storage; }
    bool operator!=(const MIROperand& rhs) { return _storage != rhs._storage; }
public:  // check function
    constexpr bool is_unused() const { return std::holds_alternative<std::monostate>(_storage); }
    constexpr bool is_imm() const { return std::holds_alternative<intmax_t>(_storage); }
    constexpr bool is_reg() const { return std::holds_alternative<MIRRegister*>(_storage); }
    constexpr bool is_reloc() const { return std::holds_alternative<MIRRelocable*>(_storage); }
    constexpr bool is_prob() const { return std::holds_alternative<double>(_storage); }
    constexpr bool is_init() const { return !std::holds_alternative<std::monostate>(_storage); }
    template <typename T>
    bool is() const { return std::holds_alternative<T>(_storage); }
public:  // gen function
    template <typename T> static MIROperand* as_imm(T val, OperandType type) { return new MIROperand(static_cast<intmax_t>(val), type); }
    static MIROperand* as_isareg(uint32_t reg, OperandType type) { return new MIROperand(new MIRRegister(reg), type); }
    static MIROperand* as_vreg(uint32_t reg, OperandType type) { return new MIROperand(new MIRRegister(reg + virtualRegBegin), type); }
    static MIROperand* as_stack_obj(uint32_t reg, OperandType type) { return new MIROperand(new MIRRegister(reg + stackObjectBegin), type); }
    static MIROperand* as_reloc(MIRRelocable* reloc) { return new MIROperand(reloc, OperandType::Special); }
    static MIROperand* as_prob(double prob) { return new MIROperand(prob, OperandType::Special); }
public:
    size_t hash() const { return std::hash<std::decay_t<decltype(_storage)>>{}(_storage); }
};

/* MIROperandHasher */
struct MIROperandHasher final {
    size_t operator()(const MIROperand* operand) const { return operand->hash(); }
};

/* MIRInst */
class MIRInst {
    static const int max_operand_num = 7;
protected:
    uint32_t _opcode;                                    // 标明指令的类型
    MIRBlock* _parent;                                   // 标明指令所在的块
    std::array<MIROperand*, max_operand_num> _operands;  // 指令操作数
public:
    MIRInst() = default;
    MIRInst(uint32_t opcode) : _opcode(opcode) {}
public:  // get function
    uint32_t opcode() const { return _opcode; }
    MIROperand* operand(int idx) const {
        assert(_operands[idx] != nullptr);
        return _operands[idx];
    }
public:  // set function
    MIRInst* set_opcode(uint32_t opcode) {
        _opcode = opcode;
        return this;
    }
    MIRInst* set_operand(int idx, MIROperand* opeand) {
        assert(idx < max_operand_num && opeand != nullptr);
        _operands[idx] = opeand;
        return this;
    }
public:
    void print(std::ostream& os);
};
using MIRInstList = std::list<MIRInst*>;

/* MIRBlock Class */
class MIRBlock : public MIRRelocable {
private:
    MIRFunction* _parent;
    std::list<MIRInst*> _insts;
    double _trip_count = 0.0;
public:
    MIRBlock() = default;
    MIRBlock(MIRFunction* parent, const std::string& name="") : MIRRelocable(name), _parent(parent) {}
public:
    void inst_sel(ir::BasicBlock* ir_bb);
    void add_inst(MIRInst* inst) { _insts.push_back(inst); }
public:  // get function
    MIRFunction* parent() { return _parent; }
    std::list<MIRInst*>& insts() { return _insts; }
    double trip_count() { return _trip_count; }
public:  // set function
    void set_trip_count(double trip_count) { _trip_count = trip_count; }
public:
    void print(std::ostream& os, CodeGenContext& ctx) override;
};

/* StackObjectUsage */
enum class StackObjectUsage {
    Argument,
    CalleeArgument,
    Local,
    RegSpill,
    CalleeSaved
};

/* StackObject */
struct StackObject final {
    uint32_t size;
    uint32_t alignment;
    int32_t offset;
    StackObjectUsage usage;
};

/* MIRJumpTable */
class MIRJumpTable final : public MIRRelocable {
private:
    std::vector<MIRRelocable*> _data;
public:
    explicit MIRJumpTable(std::string symbol) : MIRRelocable(symbol) {}
public:  // get function
    auto& data() { return _data; }
};

/* MIRFunction */
class MIRFunction : public MIRRelocable {
private:
    ir::Function* _ir_func;
    MIRModule* _parent;
    std::list<std::unique_ptr<MIRBlock>> _blocks;
    std::unordered_map<MIROperand*, StackObject> _stack_objs;
    std::vector<MIROperand*> _args;
public:
    MIRFunction();
    MIRFunction(ir::Function* ir_func, MIRModule* parent);
    MIRFunction(const std::string& name, MIRModule* parent) : MIRRelocable(name), _parent(parent) {}
public:  // get function
    auto& blocks() { return _blocks; }
    auto& args() { return _args; }
    auto& stack_objs() { return _stack_objs; }
public:  // set function
    MIROperand* add_stack_obj(uint32_t id, uint32_t size,
                              uint32_t alignment, int32_t offset,
                              StackObjectUsage usage) {
        auto ref = MIROperand::as_stack_obj(id, OperandType::Special);
        _stack_objs.emplace(ref, StackObject{size, alignment, offset, usage});
        return ref;
    }
public:  // utils function
    void print(std::ostream& os, CodeGenContext& ctx) override;
    void print_cfg(std::ostream& os);
};

/* MIRZeroStorage */
class MIRZeroStorage : public MIRRelocable {
    size_t _size;  // bytes
    bool _is_float;
public:
    MIRZeroStorage(size_t size, const std::string& name="", bool is_float=false) 
        : MIRRelocable(name), _size(size), _is_float(is_float) {}
public:
    bool is_float() const { return _is_float; }
public:
    void print(std::ostream& os, CodeGenContext& ctx) override;
};

/* MIRDataStorage */
class MIRDataStorage : public MIRRelocable {
public:
    using Storage = std::vector<uint32_t>;
private:
    Storage _data;
    bool _is_float;
    bool _readonly;
public:
    MIRDataStorage(const Storage data, bool readonly, const std::string& name="", bool is_float=false)
        : MIRRelocable(name), _data(data), _readonly(readonly), _is_float(is_float) {}
public:  // check function
    bool is_readonly() const { return _readonly; }
    bool is_float() const { return _is_float; }
public:  // set function
    uint32_t append_word(uint32_t word) {
        auto idx = static_cast<uint32_t>(_data.size());
        _data.push_back(word);
        return idx;
    }
public:
    void print(std::ostream& os, CodeGenContext& ctx) override;
};

/* MIRGlobalObject */
using MIRRelocable_UPtr = std::unique_ptr<MIRRelocable>;
struct MIRGlobalObject {
    MIRModule* parent;
    size_t align;
    MIRRelocable_UPtr reloc;  /* MIRZeroStorage OR MIRDataStorage */

    MIRGlobalObject() = default;
    MIRGlobalObject(size_t align, std::unique_ptr<MIRRelocable> reloc, MIRModule* parent)
        : parent(parent), align(align), reloc(std::move(reloc)) {}
    void print(std::ostream& os);
};

/* MIRModule */
class Target;
using MIRFunction_UPtrVec = std::vector<std::unique_ptr<MIRFunction>>;
using MIRGlobalObject_UPtrVec = std::vector<std::unique_ptr<MIRGlobalObject>>;
class MIRModule {
private:
    Target& _target;
    MIRFunction_UPtrVec _functions;
    MIRGlobalObject_UPtrVec _global_objs;
    ir::Module* _ir_module;
public:
    MIRModule() = default;
    MIRModule(ir::Module* ir_module, Target& target) : _ir_module(ir_module), _target(target) {}
public:  // get function
    MIRFunction_UPtrVec& functions() { return _functions; }
    MIRGlobalObject_UPtrVec& global_objs() { return _global_objs; }
public:
    void print(std::ostream& os);
};
}