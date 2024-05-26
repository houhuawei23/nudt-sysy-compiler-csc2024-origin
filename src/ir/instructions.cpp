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


Value* BinaryInst::getConstantRepl() {
    auto lval=get_lvalue();
    auto rval=get_rvalue();
    auto clval=dyn_cast<ir::Constant>(lval);
    auto crval=dyn_cast<ir::Constant>(rval);
    if(lval->is_i32() and rval->is_i32()){
        switch (scid())
        {
        case vADD:
            if(clval and crval)return Constant::gen_i32(clval->i32()+crval->i32());
            if(clval and clval->i32()==0)return rval;
            if(crval and crval->i32()==0)return lval;
            break;
        case vSUB:
            if(clval and crval)return Constant::gen_i32(clval->i32()-crval->i32());
            if(crval and crval->i32()==0)return lval;
            if(lval==rval)return Constant::gen_i32(0);
            break;
        case vMUL:
            if(clval and crval)return Constant::gen_i32(clval->i32()*crval->i32());
            if(clval and clval->i32()==1)return rval;
            if(crval and crval->i32()==1)return lval;
            if(clval and clval->i32()==0)return Constant::gen_i32(0);
            if(crval and crval->i32()==0)return Constant::gen_i32(0);
            break;
        case vSDIV:
            if(clval and crval)return Constant::gen_i32(clval->i32()/crval->i32());
            if(crval and crval->i32()==1)return lval;
            if(clval and clval->i32()==0)return Constant::gen_i32(0);
            if(lval==rval)return Constant::gen_i32(1);
            break;
        case vSREM:
            if(clval and crval)return Constant::gen_i32(clval->i32()%crval->i32());
            if(clval and clval->i32()==0)return Constant::gen_i32(0);
            break;
        default:
            assert(false and "Error in BinaryInst::getConstantRepl!");
            break;
        }
        return nullptr;
    }
    else if(lval->is_float32() and rval->is_float32()){
        switch (scid())
        {
        case vFADD:
            if(clval and crval)return Constant::gen_f32(clval->f32()+crval->f32());
            if(clval and clval->f32()==0) return rval;
            if(crval and crval->f32()==0) return lval;
            break;
        case vFSUB:
            if(clval and crval)return Constant::gen_f32(clval->f32()-crval->f32());
            if(crval and crval->f32()==0)return lval;
            if(lval==rval)return Constant::gen_f32(0);
            break;
        case vFMUL:
            if(clval and crval)return Constant::gen_f32(clval->f32()*crval->f32());
            if(clval and clval->f32()==1)return rval;
            if(crval and crval->f32()==1)return lval;
            if(clval and clval->f32()==0)return Constant::gen_f32(0);
            if(crval and crval->f32()==0)return Constant::gen_f32(0);
            break;
        case vFDIV:
            if(clval and crval)return Constant::gen_f32(clval->f32()/crval->f32());
            if(crval and crval->f32()==1)return lval;
            if(clval and clval->f32()==0)return Constant::gen_f32(0);
            if(lval==rval)return Constant::gen_f32(1);
            break;
        
        default:
            assert(false and "Error in BinaryInst::getConstantRepl!");
            break;
        }
        return nullptr;
    }
    return nullptr;
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
        os << get_value()->name();
        os << " to " << *type();
    }
    /* comment */
    if (not get_value()->comment().empty()) {
        os << " ; "
           << "uop " << get_value()->comment();
    }
}



Value* UnaryInst::getConstantRepl() {
    auto cval=dyn_cast<ir::Constant>(get_value());
    if(cval){
        switch (scid()) {
            case vSITOFP:
                return Constant::gen_f32(float(cval->i32()));
                break;
            case vFPTOSI:
                return Constant::gen_i32(int(cval->f32()));
                break;
            case vZEXT:
                return Constant::gen_i32(cval->i1());
                break;
            default:
                assert(false && "Invalid scid from UnaryInst::getConstantRepl");
                break;
        }
    }
    return nullptr;
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

Value* ICmpInst::getConstantRepl() {
    auto clhs = dyn_cast<Constant>(lhs());
    auto crhs = dyn_cast<Constant>(rhs());
    if(not clhs or not crhs)return nullptr;
    auto lhsval=clhs->i32();
    auto rhsval=crhs->i32();
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


Value* FCmpInst::getConstantRepl() {
    auto clhs = dyn_cast<Constant>(lhs());
    auto crhs = dyn_cast<Constant>(rhs());
    if(not clhs or not crhs)return nullptr;
    auto lhsval=clhs->f32();
    auto rhsval=crhs->f32();
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
 *      数组: <result> = getelementptr <type>, <type>* <ptrval>, i32 0, i32 <idx>
 *      指针: <result> = getelementptr <type>, <type>* <ptrval>, i32 <idx>
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

BasicBlock* PhiInst::getbbfromVal(Value* val){
    for(int i=0;i<size;i++){
        if(getval(i)==val)
            return getbb(i);
    }
    return nullptr;
}

Value* PhiInst::getvalfromBB(BasicBlock* bb){
    for(int i=0;i<size;i++){
        if(getbb(i)==bb)
            return getval(i);
    }
    return nullptr;
}


Value* PhiInst::getConstantRepl() {
    if(size==0)return nullptr;
    auto curval=getval(0);
    if(size==1)return getval(0);
    for(int i=1;i<size;i++){
        if(getval(i)!=curval)return nullptr;
    }
    return curval;
}

void PhiInst::delval(Value* val){
    int i;
    for(i=0;i<size;i++){
        if(getval(i)==val)
            break;
    }
    assert(i!=size);
    delete_operands(2*i);
    delete_operands(2*i);
    size--;
}

void PhiInst::delbb(BasicBlock* bb){
    int i;
    for(i=0;i<size;i++){
        if(getbb(i)==bb)
            break;
    }
    assert(i!=size);
    delete_operands(2*i);
    delete_operands(2*i);
    size--;
}

void PhiInst::replaceBB(BasicBlock* newBB,size_t k)
{
    set_operand(2 * k + 1, newBB);
}

/*
 * @brief: BitcastInst::print
 * @details: 
 *      <result> = bitcast <ty> <value> to i8*
 */
void BitCastInst::print(std::ostream& os) {
    os << name() << " = bitcast ";
    os << *type() << " " << value()->name();
    os << " to i8*";
}

/*
 * @brief: memset
 * @details:
 *      call void @llvm.memset.p0i8.i64(i8* <dest>, i8 0, i64 <len>, i1 false)
 */
void MemsetInst::print(std::ostream& os) {
    assert(dyn_cast<PointerType>(type()) && "type error");
    os << "call void @llvm.memset.p0i8.i64(i8* ";
    os << value()->name() << ", i8 0, i64 " << dyn_cast<PointerType>(type())->base_type()->size() << ", i1 false)";
}

}  // namespace ir

