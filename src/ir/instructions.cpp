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
    Instruction::setvarname();
    os << name() << " = alloca " << *(base_type());
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
    if (ptr()->type()->is_pointer()) os << *(ptr()->type()) << " ";
    else {
        auto ptype = ptr()->type();
        ArrayType* atype = dyn_cast<ArrayType>(ptype);
        os << *(atype->base_type()) << "* ";
    }
    os << ptr()->name();
}

void LoadInst::print(std::ostream& os) {
    Instruction::setvarname();
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
    Instruction::setvarname();
    os << name() << " = ";
    switch (scid()) {
        case vADD:
            os << "add ";
            break;
        case vFADD:
            os << "fadd ";
            break;

        case vSUB:
            os << "sub ";
            break;
        case vFSUB:
            os << "fsub ";
            break;

        case vMUL:
            os << "mul ";
            break;
        case vFMUL:
            os << "fmul ";
            break;

        case vSDIV:
            os << "sdiv ";
            break;
        case vFDIV:
            os << "fdiv ";
            break;

        case vSREM:
            os << "srem ";
            break;
        case vFREM:
            os << "frem ";
            break;

        default:
            break;
    }
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
}

/*
 * @brief Unary Instruction Output
 *      <result> = sitofp <ty> <value> to <ty2>
 *      <result> = fptosi <ty> <value> to <ty2>
 *
 *      <result> = fneg [fast-math flags]* <ty> <op1>
 */
void UnaryInst::print(std::ostream& os) {
    Instruction::setvarname();
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
            case vTRUNC:
                os << "trunc ";
                break;
            case vZEXT:
                os << "zext ";
                break;
            case vSEXT:
                os << "sext ";
                break;
            case vFPTRUNC:
                os << "fptrunc ";
                break;
            default:
                assert(false && "not valid scid");
                break;
        }
        os << *(get_value()->type()) << " ";
        os << get_value()->name() << " ";
        os << " to " << *type();
    }
}

void ICmpInst::print(std::ostream& os) {
    // <result> = icmp <cond> <ty> <op1>, <op2>   ; yields i1 or <N x i1>:result
    // %res = icmp eq i32, 1, 2
    Instruction::setvarname();
    os << name() << " = ";

    os << "icmp ";
    // cond code
    switch (scid()) {
        case vIEQ:
            os << "eq ";
            break;
        case vINE:
            os << "ne ";
            break;
        case vISGT:
            os << "sgt ";
            break;
        case vISGE:
            os << "sge ";
            break;
        case vISLT:
            os << "slt ";
            break;
        case vISLE:
            os << "sle ";
            break;
        default:
            // assert(false && "unimplemented");
            std::cerr << "Error from ICmpInst::print(), wrong Inst Type!"
                      << std::endl;
            exit(EXIT_FAILURE);
            break;
    }
    // type
    os << *lhs()->type() << " ";
    // op1
    os << lhs()->name() << ", ";
    // op2
    os << rhs()->name();
}

void FCmpInst::print(std::ostream& os) {
    // <result> = icmp <cond> <ty> <op1>, <op2>   ; yields i1 or <N x i1>:result
    // %res = icmp eq i32, 1, 2
    Instruction::setvarname();
    os << name() << " = ";

    os << "fcmp ";
    // cond code
    switch (scid()) {
        case vFOEQ:
            os << "oeq ";
            break;
        case vFONE:
            os << "one ";
            break;
        case vFOGT:
            os << "ogt ";
            break;
        case vFOGE:
            os << "oge ";
            break;
        case vFOLT:
            os << "olt ";
            break;
        case vFOLE:
            os << "ole ";
            break;
        default:
            // assert(false && "unimplemented");
            std::cerr << "Error from FCmpInst::print(), wrong Inst Type!"
                      << std::endl;
            exit(EXIT_FAILURE);
            break;
    }
    // type
    os << *lhs()->type() << " ";
    // op1
    os << lhs()->name() << ", ";
    // op2
    os << rhs()->name();
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
    } else {
        os << "label %" << dest()->name();
    }
}

/*
 * @brief: GetElementPtrInst::print
 * @details: 
 *      数组: <result> = getelementptr <type>, <type>* <ptrval>, i32 0, i32 <idx> 
 *      指针: <result> = getelementptr <type>, <type>* <ptrval>, i32 <idx>
 */
void GetElementPtrInst::print(std::ostream& os) {
    Instruction::setvarname();
    if (is_arrayInst()) {
        os << name() << " = getelementptr ";

        int dimensions = cur_dims_cnt();
        for (int i = 0; i < dimensions; i++) {
            int value = cur_dims()[i];
            os << "[" << value << " x ";
        }
        if (_id == 1) os << *(dyn_cast<ir::ArrayType>(base_type())->base_type());
        else os << *(base_type());
        for (int i = 0; i < dimensions; i++) os << "]";
        os << ", ";

        for (int i = 0; i < dimensions; i++) {
            int value = cur_dims()[i];
            os << "[" << value << " x ";
        }
        if (_id == 1) os << *(dyn_cast<ir::ArrayType>(base_type())->base_type());
        else os << *(base_type());
        for (int i = 0; i < dimensions; i++) os << "]";
        os << "* ";

        os << get_value()->name() << ", ";
        os << "i32 0, i32 " << get_index()->name();
    } else {
        os << name() << " = getelementptr " << *(base_type()) << ", " << *type() << " ";
        os << get_value()->name() << ", i32 " << get_index()->name();
    }
}

void CallInst::print(std::ostream& os) {
    if (callee()->ret_type()->is_void()) {
        os << "call ";
    } else {
        Instruction::setvarname();
        os << name() << " = call ";
    }

    // ret_type
    os << *type() << " ";
    // func name
    os << "@" << callee()->name() << "(";

    if (_rargs.size() > 0) {
        auto last = _rargs.end() - 1;  // Iterator pointing to the last element
        for (auto it = _rargs.begin(); it != last; ++it) {
            // it is a iterator, *it get the element in _rargs,
            // which is the Value* ptr
            os << *((*it)->type()) << " ";
            os << (*it)->name();
            os << ", ";
        }
        os << *((*last)->type()) << " ";
        os << (*last)->name();
    }

    os << ")";
}

void PhiInst::print(std::ostream& os){
    Instruction::setvarname();
    os << name() << " = ";
    os << "phi " << *(type()) << " ";
    // for all vals, bbs
    if(size > 0)
    {
        for(int i = 0;i <size-1;i++)
        {
            os <<"[ "<<_vals[i]->name()<<", "<<_bbs[i]->name()<<" ]"<<",";
        }
        os <<"[ "<<_vals[size-1]->name()<<", "<<_bbs[size-1]->name()<<" ]";
    }
    
}

}  // namespace ir
