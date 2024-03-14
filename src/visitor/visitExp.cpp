#include "visitor.hpp"

namespace sysy {
std::any SysYIRGenerator::visitNumberExp(SysYParser::NumberExpContext* ctx) {
    // number: ILITERAL | FLITERAL;

    ir::Value* res = nullptr;
    if (auto iLiteral = ctx->number()->ILITERAL()) {  // int
        std::string s = iLiteral->getText();
        // dec
        int base = 10;
        // hex
        if (s.length() > 2 && s[0] == '0' && (s[1] == 'x' || s[1] == 'X')) {
            base = 16;

        } else if (s[0] == '0') {  // oct
            base = 8;
        }

        res = ir::Constant::gen(std::stoi(s, 0, base));
    } else if (auto fctx = ctx->number()->FLITERAL()) {  // float
        std::string s = fctx->getText();
        float f=std::stof(s);
        // store machine code floating numbers
        unsigned int mrf=*reinterpret_cast<unsigned int*>(&f);
        std::stringstream ss;
        ss << std::hex << std::uppercase << std::setfill('0') << std::setw(8) << mrf;

        res = ir::Constant::gen(f,"0X"+ss.str()+"00000000"); // stod?
        // change to machine code when print
        // didn't realize hexadecimal floating numbers
    }
    return res;
}

std::any SysYIRGenerator::visitVarExp(SysYParser::VarExpContext* ctx){
    ir::Value* res=nullptr;
    bool isarray=not ctx->var()->LBRACKET().empty();
    std::string varname=ctx->var()->ID()->getText();
    auto valueptr=_tables.lookup(varname);
    if(valueptr==nullptr){
        std::cerr<<"Use undefined variable: \""<<varname<<"\""<<std::endl;
        exit(EXIT_FAILURE);
    }
    if(not isarray){//scalar
        res=_builder.create_load(valueptr,{},_builder.getvarname());
    }
    else{//array element
        //pass
    }
    return res;
}



}  // namespace sysy
