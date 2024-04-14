#pragma once
#include <array>
#include <list>
#include <variant>
#include <vector>
#include "ir/ir.hpp"

namespace mir {
class MIRRelocable;
class MIRModule;
class MIRFunction;  // public MIRRelocable
class MIRBlock;     // public MIRRelocable
class MIRInst;
class MIRRegister;
class MIROperand;
class MIRGlobalObject;
class MIRZeroStorage;
class MIRDataStorage;
struct StackObject;
// class MIR

class MIRRelocable {
    std::string _name;

   public:
    MIRRelocable(const std::string& name = "") : _name(name) {}
    virtual ~MIRRelocable() = default;
    auto name() const { return _name; }

    void print(std::ostream& os);  // as dump
};

constexpr uint32_t virtualRegBegin = 0b0101U << 28;
constexpr uint32_t stackObjectBegin = 0b1010U << 28;
constexpr uint32_t invalidReg = 0b1100U << 28;
constexpr bool isISAReg(uint32_t x) {
    return x < virtualRegBegin;
}
constexpr bool isVirtualReg(uint32_t x) {
    return (x & virtualRegBegin) == virtualRegBegin;
}
constexpr bool isStackObject(uint32_t x) {
    return (x & stackObjectBegin) == stackObjectBegin;
}

enum class OperandType : uint32_t {
    Bool,
    Int8,
    Int16,
    Int32,
    Int64,
    Float32,
    Special,
    // %hi/%lo for relocatable addresses
    HighBits,
    LowBits,
};

class MIRRegister {
    uint32_t _reg;
    // MIRRegisterFlag flag =;
   public:
    MIRRegister() = default;
    MIRRegister(uint32_t reg) : _reg(reg) {}

   private:
    // RegType _type;

   public:
    bool operator==(const MIRRegister& rhs) const { return _reg == rhs._reg; }

    bool operator!=(const MIRRegister& rhs) const { return _reg != rhs._reg; }

    uint32_t reg() { return _reg; }

   public:
    void print(std::ostream& os);
};

enum MIRGenericInst : uint32_t {
    InstJump,
    InstBranch,
    InstUnreachable,
    // memory
    InstLoad,
    InstStore,
    // arth
    InstAdd,
    InstSub,
    InstMul,
    InstUDiv,
    InstURem,
    // bitwise
    InstAnd,
    InstOr,
    InstXor,
    InstLShr,  // logic shift right
    InstAShr,  // arth shift right
    // Signed div/rem
    InstSDiv,
    InstSRem,
    // minmax
    InstSMin,
    InstSMax,
    // unary
    InstNeg,
    InstAbs,
    // fp
    InstFAdd,
    InstFSub,
    InstFMul,
    InstFDiv,
    InstFNeg,
    InstFAbs,
    InstFFma,
    // comp
    InstICmp,
    InstFCmp,
    // conversion
    InstSExt,
    InstZExt,
    InstTrunc,
    InstF2U,
    InstF2S,
    InstU2F,
    InstS2F,
    InstFCast,
    // misc
    InstCopy,
    InstSelect,
    InstLoadGlobalAddress,
    InstLoadImm,
    InstLoadStackObjectAddr,
    InstCopyFromReg,
    InstCopyToReg,
    InstLoadImmToReg,
    InstLoadRegFromStack,
    InstStoreRegToStack,

    // hhw add
    InstRet,

    // ISA specific
    ISASpecificBegin,

};

class MIROperand {
   private:
    std::variant<std::monostate, MIRRegister*, MIRRelocable*, intmax_t, double>
        _storage{std::monostate{}};
    OperandType _type = OperandType::Special;
    // int tmp;
    // union _operand {
    //     MIRRegister* reg;
    //     // int i;
    //     // float f;
    // };
   public:
    MIROperand() = default;
    template <typename T>
    MIROperand(T x, OperandType type) : _storage(x), _type(type) {}

    auto& storage() { return _storage; }
    auto type() { return _type; }

    bool operator==(const MIROperand& rhs) { return _storage == rhs._storage; }
    bool operator!=(const MIROperand& rhs) { return _storage != rhs._storage; }

    // hash?

    intmax_t imm() { return std::get<intmax_t>(_storage); }
    uint32_t reg() { return std::get<MIRRegister*>(_storage)->reg(); }
    MIRRelocable* reloc() { return std::get<MIRRelocable*>(_storage); }

    bool is_imm() { return std::holds_alternative<intmax_t>(_storage); }
    bool is_reg() { return std::holds_alternative<MIRRegister*>(_storage); }
    bool is_reloc() { return std::holds_alternative<MIRRelocable*>(_storage); }
    bool is_prob() { return false; }
    bool is_init() { return !std::holds_alternative<std::monostate>(_storage); }

    template <typename T>
    bool is() {
        return std::holds_alternative<T>(_storage);
    }

    // gen
    template <typename T>
    static MIROperand* as_imm(T val, OperandType type) {
        return new MIROperand(static_cast<intmax_t>(val), type);
    }

    // template <typename T>
    static MIROperand* as_isareg(uint32_t reg, OperandType type) {
        // assert is isa reg
        auto reg_obj = new MIRRegister(reg);
        auto operand = new MIROperand(reg_obj, type);
        return operand;
    }

    // as vreg
    static MIROperand* as_vreg(uint32_t reg, OperandType type) {
        return new MIROperand(new MIRRegister(reg + virtualRegBegin), type);
    }
    // as stack obj
    static MIROperand* as_stack_obj(uint32_t reg, OperandType type) {
        return new MIROperand(new MIRRegister(reg + stackObjectBegin), type);
    }
    // as invalid reg
    // as reloc
    // as prob?

   public:
    // void print(std::ostream& os);
};

// addi	sp,sp,-16
// sd	s0,8(sp)
// addi	s0,sp,16
class MIRInst {
    static const int max_operand_num = 7;

   protected:
    uint32_t _code;
    MIRBlock* _parent;
    std::array<MIROperand*, max_operand_num> _operands;
    // std::vector<MIROperand*> _operands;
    // std::vector<MIROperand*> _operands(max_operand_num);
   public:
    MIRInst() = default;
    // MIRInst(ir::Instruction* ir_inst, MIRBlock* parent);
    MIRInst(uint32_t code) : _code(code) {}

    // get
    uint32_t code() { return _code; }
    MIROperand* operand(int idx) { return _operands[idx]; }

    // set
    MIRInst& set_operand(int idx, MIROperand* opeand) {
        assert(idx < max_operand_num);
        _operands[idx] = opeand;
        return *this;
    }

   public:
    void print(std::ostream& os);
};

class MIRBlock : public MIRRelocable {
   private:
    MIRFunction* _parent;
    std::list<MIRInst*> _insts;
    ir::BasicBlock* _ir_block;

   public:
    MIRBlock() = default;
    MIRBlock(ir::BasicBlock* ir_block, MIRFunction* parent)
        : _ir_block(ir_block), _parent(parent) {}

    void inst_sel(ir::BasicBlock* ir_bb);
    void add_inst(MIRInst* inst) { _insts.push_back(inst); }

   public:
    void print(std::ostream& os);
};

enum class StackObjectUsage {
    Argument,
    CalleeArgument,
    Local,
    RegSpill,
    CalleeSaved
};

struct StackObject final {
    uint32_t size;
    uint32_t alignment;
    int32_t offset;  // positive
    StackObjectUsage usage;
};

class MIRFunction : public MIRRelocable {
   private:
    MIRModule* _parent;
    std::vector<MIRBlock*> _blocks;
    ir::Function* _ir_func;
    std::unordered_map<MIROperand*, StackObject*> _stack_objs;
    std::vector<MIROperand*> _args;
    // std::string _name;

   public:
    MIRFunction();
    MIRFunction(ir::Function* ir_func, MIRModule* parent);
    MIRFunction(const std::string& name, MIRModule* parent)
        : MIRRelocable(name), _parent(parent) {}
    // MIROperand* add_stack_obj()
    auto& blocks() { return _blocks; }
    auto& args() { return _args; }
    auto& stack_objs() { return _stack_objs; }

    MIROperand* add_stack_obj(uint32_t id,
                              uint32_t size,
                              uint32_t alignment,
                              int32_t offset,
                              StackObjectUsage usage) {
        auto ref = MIROperand::as_stack_obj(id, OperandType::Special);
        // _stack_objs.emplace(ref, new StackObject(size, alignment, offset,
        // usage));
        return ref;
    }

   public:
    void print(std::ostream& os);
    void print_cfg(std::ostream& os);
};

// all zero storage
class MIRZeroStorage : public MIRRelocable {
    size_t _size;  // bytes

   public:
    MIRZeroStorage(size_t size, const std::string& name = "")
        : MIRRelocable(name), _size(size) {}

    void print(std::ostream& os);
};

// data storage
class MIRDataStorage : public MIRRelocable {
   public:
    using Storage = std::vector<uint32_t>;  // words vector

   private:
    Storage _data;
    bool _readonly;

   public:
    MIRDataStorage(const Storage data,
                   bool readonly,
                   const std::string& name = "")
        : MIRRelocable(name), _data(data), _readonly(readonly) {}

    bool is_ro() const { return _readonly; }

    uint32_t append_word(uint32_t word) {
        auto idx = static_cast<uint32_t>(_data.size());
        _data.push_back(word);
        return idx;  // idx of the last word
    }

    void print(std::ostream& os);
};

/*
 * in cmmc, MIRGlobal contains MIRFunction/MIRDataStorage/MIRZeroStorage
 */
using MIRRelocable_UPtr = std::unique_ptr<MIRRelocable>;
class MIRGlobalObject {
   private:
    MIRModule* _parent;
    ir::Value* _ir_global;
    size_t align;
    MIRRelocable_UPtr _reloc;  // MIRZeroStorage, MIRDataStorage

   public:
    MIRGlobalObject() = default;
    // MIRGlobalObject(ir::Value* ir_global, MIRModule* parent)
    //     : _parent(parent), _ir_global(ir_global) {
    //     std::string var_name = _ir_global->name();
    //     var_name = var_name.substr(1, var_name.length() - 1);
    //     _ir_global->set_name(var_name);
    // }
    MIRGlobalObject(size_t align, std::unique_ptr<MIRRelocable> reloc, MIRModule* parent)
        : _parent(parent), align(align), _reloc(std::move(reloc)) {}

   public:
    void print(std::ostream& os);
};

class Target;
using MIRFunction_UPtrVec = std::vector<std::unique_ptr<MIRFunction>>;
using MIRGlobalObject_UPtrVec = std::vector<std::unique_ptr<MIRGlobalObject>>;

class MIRModule {
   private:
    Target& _target;
    MIRFunction_UPtrVec _functions;
    MIRGlobalObject_UPtrVec _global_objs;
    // std::vector<MIRGlobalObject*> _global_objs;
    ir::Module* _ir_module;

   public:
    MIRModule() = default;
    MIRModule(ir::Module* ir_module, Target& target)
        : _ir_module(ir_module), _target(target) {}

    MIRFunction_UPtrVec& functions() { return _functions; }
    MIRGlobalObject_UPtrVec& global_objs() { return _global_objs; }

   public:
    void print(std::ostream& os);
};

}  // namespace mir
