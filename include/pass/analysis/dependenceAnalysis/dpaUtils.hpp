#pragma once
#include "ir/ir.hpp"
#include "pass/pass.hpp"
#include "pass/analysis/dependenceAnalysis/DependenceAnalysis.hpp"

namespace pass{
    class LoopDependenceInfo;
    struct gepIdx;
    enum baseAddrType{
        globalType,
        localType,
        argType
    };
    enum idxType{//单个idx的类型
        iLOOPINVARIANT,//lpI
        iIDV,//i
        iIDVPLUSMINUSFORMULA,//i+a i-a (a==lpI)
        iSIV,//k*i+-a (a==lpI)
        iCALL,//f()
        iLOAD,//a[] or global
        iELSE,
    };
    struct gepIdx{
        std::vector<ir::Value*>idxList;
        std::map<ir::Value*,idxType>idxTypes;
    };
    class LoopDependenceInfo{
        private:
            //for outside info
            ir::Loop* parent;//当前循环
            bool isParallel;//是否应当并行
            //for baseaddr
            std::set<ir::Value*>baseAddrs;//存储该循环中用到的基址
            std::map<ir::Value*,bool>baseAddrIsRead;//当前基址是否读
            std::map<ir::Value*,bool>baseAddrIsWrite;//当前基址是否写
            bool isBaseAddrPossiblySame;//基址别名可能
            //for subAddr
            std::map<ir::Value*,std::set<ir::GetElementPtrInst*>>baseAddrToSubAddrs;//子地址和基地址映射
            std::map<ir::GetElementPtrInst*,gepIdx*>subAddrToGepIdx;//保存gep的Idx信息
            std::map<ir::GetElementPtrInst*,bool>subAddrIsRead;//当前子地址是否读
            std::map<ir::GetElementPtrInst*,bool>subAddrIsWrite;//当前子地址是否写
            //for readwrite
            std::map<ir::GetElementPtrInst*,std::set<ir::Instruction*>>subAddrToInst;//子地址到存取语句
            std::set<ir::Instruction*>memInsts;//在当前这个循环中进行了存取的语句

        public:
            void makeLoopDepInfo(ir::Loop* lp);//直接根据循环相关信息对当前info进行构建
            void clearAll();
            bool getIsParallel(){return isParallel;}
            bool getIsBaseAddrPossiblySame(){return isBaseAddrPossiblySame;}
        private:
            void addPtr(ir::Value* val,ir::Instruction* inst);//用于添加一个指针进入
    };
    ir::Value* getBaseAddr(ir::Value* val);
    baseAddrType getBaseAddrType(ir::Value* val);
    bool isTwoBaseAddrPossiblySame(ir::Value* ptr1,ir::Value* ptr2);
};

