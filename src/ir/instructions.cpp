#include "ir/instructions.hpp"
#include "ir/constant.hpp"
#include "ir/utils_ir.hpp"

namespace ir {
//! Value <- User <- Instruction <- XxxInst

/*
 * @brief AllocaInst::print
 * @details:
 *      alloca <ty>
 */
void AllocaInst::print(std::ostream& os) {
    os << name() << " = alloca " << *(base_type());
    if (not _comment.empty()) {
        os << " ; " << _comment << "*";
    }
}

/*
 * @brief StoreInst::print
 * @details:
 *      store <ty> <value>, ptr <pointer>
 * @note:
 *      ptr: ArrayType or PointerType
 */
void StoreInst::print(std::ostream& os) {
    os << "store ";
    os << *(value()->type()) << " ";
    if (ir::isa<ir::Constant>(value())) {
        auto v = dyn_cast<ir::Constant>(value());
        os << *(value()) << ", ";
    } else {
        os << value()->name() << ", ";
    }
    if (ptr()->type()->is_pointer())
        os << *(ptr()->type()) << " ";
    else {
        auto ptype = ptr()->type();
        ArrayType* atype = dyn_cast<ArrayType>(ptype);
        os << *(atype->base_type()) << "* ";
    }
    os << ptr()->name();
}

void LoadInst::print(std::ostream& os) {
    os << name() << " = load ";
    auto ptype = ptr()->type();
    if (ptype->is_pointer()) {
        os << *dyn_cast<PointerType>(ptype)->base_type() << ", ";
        os << *ptype << " ";
    } else {
        os << *dyn_cast<ArrayType>(ptype)->base_type() << ", ";
        os << *dyn_cast<ArrayType>(ptype)->base_type() << "* ";
    }
    os << ptr()->name();

    // comment
    if (not _comment.empty()) {
        os << " ; "
           << "load " << _comment;
    }
}

void ReturnInst::print(std::ostream& os) {
    os << "ret ";
    auto ret = return_value();
    if (ret) {
        os << *ret->type() << " ";
        if (ir::isa<ir::Constant>(ret))
            os << *(ret);
        else
            os << ret->name();
    } else {
        os << "void";
    }
}

/*
 * @brief Binary Instruction Output
 *      <result> = add <ty> <op1>, <op2>
 */
void BinaryInst::print(std::ostream& os) {
    os << name() << " = ";
    auto opstr = [scid = scid()] {
        switch (scid) {
            case vADD:
                return "add";
            case vFADD:
                return "fadd";
            case vSUB:
                return "sub";
            case vFSUB:
                return "fsub";
            case vMUL:
                return "mul";
            case vFMUL:
                return "fmul";
            case vSDIV:
                return "sdiv";
            case vFDIV:
                return "fdiv";
            case vSREM:
                return "srem";
            case vFREM:
                return "frem";
            default:
                return "unknown";
        }
    }();
    os << opstr << " ";
    // <type>
    os << *type() << " ";
    // <op1>
    if (ir::isa<ir::Constant>(get_lvalue()))
        os << *(get_lvalue()) << ", ";
    else
        os << get_lvalue()->name() << ", ";
    // <op2>
    if (ir::isa<ir::Constant>(get_rvalue()))
        os << *(get_rvalue());
    else
        os << get_rvalue()->name();

    /* comment */
    if (not get_lvalue()->comment().empty() &&
        not get_rvalue()->comment().empty()) {
        os << " ; " << get_lvalue()->comment();
        os << " " << opstr << " ";
        os << get_rvalue()->comment();
    }
}

bool BinaryInst::is_constprop() {
    auto lconst = dyn_cast<Constant>(get_lvalue());
    auto rconst = dyn_cast<Constant>(get_rvalue());
    return lconst and rconst;
}

Constant* BinaryInst::getConstantRepl() {
    auto lval = dyn_cast<Constant>(get_lvalue());
    auto rval = dyn_cast<Constant>(get_rvalue());
    if (get_lvalue()->is_i32()) {
        auto lvali32 = lval->i32();
        auto rvali32 = rval->i32();
        switch (scid()) {
            case vADD:
                return Constant::gen_i32(lvali32 + rvali32);
            case vSUB:
                return Constant::gen_i32(lvali32 - rvali32);
            case vMUL:
                return Constant::gen_i32(lvali32 * rvali32);
            case vSDIV:
                return Constant::gen_i32(lvali32 / rvali32);
            case vSREM:
                return Constant::gen_i32(lvali32 % rvali32);
            default:
                assert(false and "Error in BinaryInst::getConstantRepl");
                break;
        }
    } else if (get_lvalue()->is_float32()) {
        auto lvalf32 = lval->f32();
        auto rvalf32 = rval->f32();
        switch (scid()) {
            case vFADD:
                return Constant::gen_f32(lvalf32 + rvalf32);
            case vFSUB:
                return Constant::gen_f32(lvalf32 - rvalf32);
            case vFMUL:
                return Constant::gen_f32(lvalf32 * rvalf32);
            case vFDIV:
                return Constant::gen_f32(lvalf32 / rvalf32);
            default:
                assert(false and "Error in BinaryInst::getConstantRepl");
                break;
        }
    } else {
        assert(false &&
               "Not implemented type of binary inst const propagation");
    }
}



/*
 * @brief Unary Instruction Output
 *      <result> = sitofp <ty> <value> to <ty2>
 *      <result> = fptosi <ty> <value> to <ty2>
 *
 *      <result> = fneg [fast-math flags]* <ty> <op1>
 */
void UnaryInst::print(std::ostream& os) {
    os << name() << " = ";
    if (scid() == vFNEG) {
        os << "fneg " << *type() << " " << get_value()->name();
    } else {
        switch (scid()) {
            case vSITOFP:
                os << "sitofp ";
                break;
            case vFPTOSI:
                os << "fptosi ";
                break;
            case vTRUNC:  // not used
                os << "trunc ";
                break;
            case vZEXT:
                os << "zext ";
                break;
            case vSEXT:  // not used
                os << "sext ";
                break;
            case vFPTRUNC:
                os << "fptrunc ";  // not used
                break;
            default:
                assert(false && "not valid scid");
                break;
        }
        os << *(get_value()->type()) << " ";
        os << get_value()->name() << " ";
        os << " to " << *type();
    }
    /* comment */
    if (not get_value()->comment().empty()) {
        os << " ; "
           << "uop " << get_value()->comment();
    }
}

bool UnaryInst::is_constprop() {
    return dyn_cast<Constant>(get_value());
}

Constant* UnaryInst::getConstantRepl() {
    auto cval = dyn_cast<Constant>(get_value());
    switch (scid()) {
        case vSITOFP:
            return Constant::gen_f32(float(cval->i32()));
        case vFPTOSI:
            return Constant::gen_i32(int(cval->f32()));
        case vZEXT:  // only i1->i32
            assert(cval->is_i1() and "ZEXT must be i1 -> i32");
            return Constant::gen_i32(cval->i1() ? 1 : 0);
        default:
            assert(false and "unary const flod error");
    }
}


void ICmpInst::print(std::ostream& os) {
    // <result> = icmp <cond> <ty> <op1>, <op2>   ; yields i1 or <N x i1>:result
    // %res = icmp eq i32, 1, 2
    os << name() << " = ";

    os << "icmp ";

    auto cmpstr = [scid = scid()] {
        switch (scid) {
            case vIEQ:
                return "eq";
            case vINE:
                return "ne";
            case vISGT:
                return "sgt";
            case vISGE:
                return "sge";
            case vISLT:
                return "slt";
            case vISLE:
                return "sle";
            default:
                // assert(false && "unimplemented");
                std::cerr << "Error from ICmpInst::print(), wrong Inst Type!"
                          << std::endl;
                return "unknown";
        }
    }();

    // cond code
    os << cmpstr << " ";
    // type
    os << *lhs()->type() << " ";
    // op1
    os << lhs()->name() << ", ";
    // op2
    os << rhs()->name();
    /* comment */
    if (not lhs()->comment().empty() && not rhs()->comment().empty()) {
        os << " ; " << lhs()->comment() << " " << cmpstr << " "
           << rhs()->comment();
    }
}
bool ICmpInst::is_constprop() {
    auto lconst = dyn_cast<Constant>(lhs());
    auto rconst = dyn_cast<Constant>(rhs());
    return lconst and rconst;
}

Constant* ICmpInst::getConstantRepl() {
    auto lhsval = dyn_cast<Constant>(lhs())->i32();
    auto rhsval = dyn_cast<Constant>(rhs())->i32();
    switch (scid()) {
        case vIEQ:
            return Constant::gen_i1(lhsval == rhsval);
        case vINE:
            return Constant::gen_i1(lhsval != rhsval);
        case vISGT:
            return Constant::gen_i1(lhsval > rhsval);
        case vISLT:
            return Constant::gen_i1(lhsval < rhsval);
        case vISGE:
            return Constant::gen_i1(lhsval >= rhsval);
        case vISLE:
            return Constant::gen_i1(lhsval <= rhsval);
        default:
            assert(false and "icmpinst const flod error");
    }
}

void FCmpInst::print(std::ostream& os) {
    // <result> = icmp <cond> <ty> <op1>, <op2>   ; yields i1 or <N x i1>:result
    // %res = icmp eq i32, 1, 2
    os << name() << " = ";

    os << "fcmp ";
    // cond code
    auto cmpstr = [scid = scid()] {
        switch (scid) {
            case vFOEQ:
                return "oeq";
            case vFONE:
                return "one";
            case vFOGT:
                return "ogt";
            case vFOGE:
                return "oge";
            case vFOLT:
                return "olt";
            case vFOLE:
                return "ole";
            default:
                // assert(false && "unimplemented");
                std::cerr << "Error from FCmpInst::print(), wrong Inst Type!"
                          << std::endl;
                return "unknown";
        }
    }();
    os << cmpstr << " ";
    // type
    os << *lhs()->type() << " ";
    // op1
    os << lhs()->name() << ", ";
    // op2
    os << rhs()->name();
    /* comment */
    if (not lhs()->comment().empty() && not rhs()->comment().empty()) {
        os << " ; " << lhs()->comment() << " " << cmpstr << " "
           << rhs()->comment();
    }
}

bool FCmpInst::is_constprop() {
    auto lconst = dyn_cast<Constant>(lhs());
    auto rconst = dyn_cast<Constant>(rhs());
    return lconst and rconst;
}

Constant* FCmpInst::getConstantRepl() {
    auto lhsval = dyn_cast<Constant>(lhs())->f32();
    auto rhsval = dyn_cast<Constant>(rhs())->f32();
    switch (scid()) {
        case vFOEQ:
            return Constant::gen_i1(lhsval == rhsval);
        case vFONE:
            return Constant::gen_i1(lhsval != rhsval);
        case vFOGT:
            return Constant::gen_i1(lhsval > rhsval);
        case vFOLT:
            return Constant::gen_i1(lhsval < rhsval);
        case vFOGE:
            return Constant::gen_i1(lhsval >= rhsval);
        case vFOLE:
            return Constant::gen_i1(lhsval <= rhsval);
        default:
            assert(false and "fcmpinst const flod error");
    }
}


/*
 * @brief: BranchInst::print
 * @details:
 *      br i1 <cond>, label <iftrue>, label <iffalse>
 *      br label <dest>
 */
void BranchInst::print(std::ostream& os) {
    os << "br ";
    if (is_cond()) {
        os << "i1 ";
        os << cond()->name() << ", ";
        os << "label %" << iftrue()->name() << ", ";
        os << "label %" << iffalse()->name();
        /* comment */
        if (not iftrue()->comment().empty() &&
            not iffalse()->comment().empty()) {
            os << " ; "
               << "br " << iftrue()->comment() << ", " << iffalse()->comment();
        }

    } else {
        os << "label %" << dest()->name();
        /* comment */
        if (not dest()->comment().empty()) {
            os << " ; "
               << "br " << dest()->comment();
        }
    }
}

/*
 * @brief: GetElementPtrInst::print
 * @details:
 *      数组: <result> = getelementptr <type>, <type>* <ptrval>, i32 0, i32
 * <idx> 指针: <result> = getelementptr <type>, <type>* <ptrval>, i32 <idx>
 */
void GetElementPtrInst::print(std::ostream& os) {
    if (is_arrayInst()) {
        os << name() << " = getelementptr ";

        int dimensions = cur_dims_cnt();
        for (int i = 0; i < dimensions; i++) {
            int value = cur_dims()[i];
            os << "[" << value << " x ";
        }
        if (_id == 1)
            os << *(dyn_cast<ir::ArrayType>(base_type())->base_type());
        else
            os << *(base_type());
        for (int i = 0; i < dimensions; i++)
            os << "]";
        os << ", ";

        for (int i = 0; i < dimensions; i++) {
            int value = cur_dims()[i];
            os << "[" << value << " x ";
        }
        if (_id == 1)
            os << *(dyn_cast<ir::ArrayType>(base_type())->base_type());
        else
            os << *(base_type());
        for (int i = 0; i < dimensions; i++)
            os << "]";
        os << "* ";

        os << get_value()->name() << ", ";
        os << "i32 0, i32 " << get_index()->name();
    } else {
        os << name() << " = getelementptr " << *(base_type()) << ", " << *type()
           << " ";
        os << get_value()->name() << ", i32 " << get_index()->name();
    }
}

void CallInst::print(std::ostream& os) {
    if (callee()->ret_type()->is_void()) {
        os << "call ";
    } else {
        os << name() << " = call ";
    }

    // ret_type
    os << *type() << " ";
    // func name
    os << "@" << callee()->name() << "(";

    if (operands().size() > 0) {
        // Iterator pointing to the last element
        auto last = operands().end() - 1;
        for (auto it = operands().begin(); it != last; ++it) {
            // it is a iterator, *it get the element in operands,
            // which is the Value* ptr
            auto val = (*it)->value();
            os << *(val->type()) << " ";
            os << val->name();
            os << ", ";
        }
        auto lastval = (*last)->value();
        os << *(lastval->type()) << " ";
        os << lastval->name();
    }
    os << ")";
}

void PhiInst::print(std::ostream& os) {
    // 在打印的时候对其vals和bbs进行更新
    os << name() << " = ";
    os << "phi " << *(type()) << " ";
    // for all vals, bbs
    for (int i = 0; i < size; i++) {
        os << "[ " << getval(i)->name() << ", %" << getbb(i)->name() << " ]";
        if (i != size - 1)
            os << ",";
    }
}

bool PhiInst::is_constprop() {
    auto cur_val = dyn_cast<Constant>(getval(0));
    for (int i = 0; i < size; i++) {
        auto cval = dyn_cast<Constant>(getval(i));
        if (not cval)
            return false;
        if (not cur_val->isequal(cval))
            return false;
    }

    return true;
}

Constant* PhiInst::getConstantRepl() {
    return dyn_cast<Constant>(getval(0));
}

}  // namespace ir
