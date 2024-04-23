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

struct CodeGenContext;

class MIRRelocable {
    std::string _name;

   public:
    MIRRelocable(const std::string& name = "") : _name(name) {}
    virtual ~MIRRelocable() = default;
    auto name() const { return _name; }

    virtual void print(std::ostream& os, CodeGenContext& ctx) = 0;  // as dump
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

constexpr uint32_t getOperandSize(const OperandType type) {
    switch (type) {
        case OperandType::Int8:
            return 1;
        case OperandType::Int16:
            return 2;
        case OperandType::Int32:
            return 4;
        case OperandType::Int64:
            return 8;
        case OperandType::Float32:
            return 4;
        default:
            return 0;  // unsupported
    }
}

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
    InstShl,
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
    InstLoadStackObjectAddr,  // 43
    InstCopyFromReg,
    InstCopyToReg,
    InstLoadImmToReg,
    InstLoadRegFromStack,
    InstStoreRegToStack,

    // hhw add
    InstReturn,

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
    auto& getStorage() const noexcept { return _storage; }
    intmax_t imm() { return std::get<intmax_t>(_storage); }
    uint32_t reg() const { return std::get<MIRRegister*>(_storage)->reg(); }
    MIRRelocable* reloc() { return std::get<MIRRelocable*>(_storage); }

    constexpr bool is_imm() {
        return std::holds_alternative<intmax_t>(_storage);
    }
    constexpr bool is_reg() {
        return std::holds_alternative<MIRRegister*>(_storage);
    }
    constexpr bool is_reloc() { return std::holds_alternative<MIRRelocable*>(_storage); }
    constexpr bool is_prob() { return false; }
    constexpr bool is_init() {
        return !std::holds_alternative<std::monostate>(_storage);
    }

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
    static MIROperand* as_reloc(MIRRelocable* reloc) {
        return new MIROperand(reloc, OperandType::Special);
    }
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
    uint32_t _opcode;
    MIRBlock* _parent;
    std::array<MIROperand*, max_operand_num> _operands;
    // std::vector<MIROperand*> _operands;
    // std::vector<MIROperand*> _operands(max_operand_num);
   public:
    MIRInst() = default;
    // MIRInst(ir::Instruction* ir_inst, MIRBlock* parent);
    MIRInst(uint32_t opcode) : _opcode(opcode) {}

    // get
    uint32_t opcode() { return _opcode; }
    MIROperand* operand(int idx) {
        assert(_operands[idx] != nullptr);
        return _operands[idx];
    }

    // set
    MIRInst* set_operand(int idx, MIROperand* opeand) {
        assert(idx < max_operand_num);
        assert(opeand != nullptr);
        _operands[idx] = opeand;
        return this;
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
    MIRBlock(ir::BasicBlock* ir_block,
             MIRFunction* parent,
             const std::string& name = "")
        : MIRRelocable(name), _ir_block(ir_block), _parent(parent) {}

    void inst_sel(ir::BasicBlock* ir_bb);
    void add_inst(MIRInst* inst) { _insts.push_back(inst); }

    std::list<MIRInst*>& insts() { return _insts; }
    ir::BasicBlock* ir_block() { return _ir_block; }

   public:
    void print(std::ostream& os, CodeGenContext& ctx) override;
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
    ir::Function* _ir_func;

    MIRModule* _parent;
    // std::vector<MIRBlock*> _blocks;
    std::list<std::unique_ptr<MIRBlock>> _blocks;
    std::unordered_map<MIROperand*, StackObject> _stack_objs;
    std::vector<MIROperand*> _args;

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
        _stack_objs.emplace(ref, StackObject{size, alignment, offset, usage});
        return ref;
    }

   public:
    void print(std::ostream& os, CodeGenContext& ctx) override;
    void print_cfg(std::ostream& os);
};

// all zero storage
class MIRZeroStorage : public MIRRelocable {
    size_t _size;  // bytes

   public:
    MIRZeroStorage(size_t size, const std::string& name = "")
        : MIRRelocable(name), _size(size) {}

    void print(std::ostream& os, CodeGenContext& ctx) override;
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

    bool is_readonly() const { return _readonly; }

    uint32_t append_word(uint32_t word) {
        auto idx = static_cast<uint32_t>(_data.size());
        _data.push_back(word);
        return idx;  // idx of the last word
    }

    void print(std::ostream& os, CodeGenContext& ctx) override;
};

/*
 * in cmmc, MIRGlobal contains MIRFunction/MIRDataStorage/MIRZeroStorage
 */
using MIRRelocable_UPtr = std::unique_ptr<MIRRelocable>;
class MIRGlobalObject {
   public:
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
    MIRGlobalObject(size_t align,
                    std::unique_ptr<MIRRelocable> reloc,
                    MIRModule* parent)
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
