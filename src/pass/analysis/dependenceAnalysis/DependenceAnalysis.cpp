#include "pass/analysis/dependenceAnalysis/DependenceAnalysis.hpp"
#include "pass/analysis/dependenceAnalysis/dpaUtils.hpp"
using namespace pass;

void DependenceAnalysis::run(ir::Function* func, TopAnalysisInfoManager* tp) {
  std::cerr << "da(" << func->name() << ") ";
  topmana = tp;
  std::cerr << "1 ";
  domctx = tp->getDomTree(func);
  std::cerr << "2 ";
  lpctx = tp->getLoopInfo(func);
  std::cerr << "3 ";
  idvctx = tp->getIndVarInfo(func);
  std::cerr << "4 ";
  sectx = tp->getSideEffectInfo();
  std::cerr << "5 ";
  cgctx = tp->getCallGraph();
  std::cerr << "6 ";
  dpctx = tp->getDepInfoWithoutRefresh(func);
  std::cerr << "7 ";
  func->rename();

  for (auto lp : lpctx->loops()) {  // 从最上层的循环开始遍历
    if (lp->parentloop() == nullptr) runOnLoop(lp);
  }
}

void DependenceAnalysis::runOnLoop(ir::Loop* lp) {
  if (lp == nullptr) {
    std::cerr << "rLp(nullptr) ";
    return;
  }
  std::cerr << "rLp(" << lp->header()->name() << ") ";
  for (auto subLp : lp->subLoops()) {  // 先处理子循环
    runOnLoop(subLp);
  }
  auto depInfoForLpPtr = dpctx->getLoopDependenceInfo(lp);
  if (depInfoForLpPtr == nullptr) {  // 取出dpctx中的LoopDependenceInfo
    depInfoForLpPtr = new LoopDependenceInfo();
    dpctx->setDepInfoLp(lp, depInfoForLpPtr);
  }
  auto depInfoForLp = depInfoForLpPtr;
  depInfoForLp->clearAll();
  depInfoForLp->setTp(topmana);
  auto func = lp->header()->function();
  for (auto subLp : lp->subLoops()) {  // 将子循环的信息合并到上层
    auto subLoopDepInfo = (LoopDependenceInfo*)(dpctx->getLoopDependenceInfo(subLp));
    depInfoForLp->getInfoFromSubLoop(subLp, subLoopDepInfo);
  }
  // 分析所有的inst
  depInfoForLp->makeLoopDepInfo(lp, topmana);
  // 别名分析测试
  bool isSame = false;
  for (auto setIter = depInfoForLp->getBaseAddrs().begin();
       setIter != depInfoForLp->getBaseAddrs().end(); setIter++) {
    for (auto setIter2 = depInfoForLp->getBaseAddrs().begin(); setIter2 != setIter; setIter2++) {
      if (setIter2 == setIter) continue;
      if (isTwoBaseAddrPossiblySame(*setIter, *setIter2, func, cgctx, topmana)) {
        isSame = true;
        break;
      }
    }
  }

  if (isSame) {
    depInfoForLp->setIsBaseAddrPossiblySame(isSame);
    depInfoForLp->setIsParallel(false);
    std::cerr << "Alias!" << std::endl;
    return;
  }
  // 为并行设计的依赖关系分析
  //  depInfoForLp->print(std::cerr);
  bool isParallel = true;
  auto defaultIdv = idvctx->getIndvar(lp);
  for (auto bd : depInfoForLp->getBaseAddrs()) {
    auto& subAddrs = depInfoForLp->baseAddrToSubAddrSet(bd);
    for (auto subAd : subAddrs) {
      auto gepidx = depInfoForLp->getGepIdx(subAd);
      makeGepIdx(lp, defaultIdv, gepidx);
    }
  }

  // depInfoForLp->print(std::cerr);
  // 进行针对并行化的依赖关系
  for (auto bd : depInfoForLp->getBaseAddrs()) {
    auto& subAddrs = depInfoForLp->baseAddrToSubAddrSet(bd);
    // 要么只有一个子地址，要么只有读，就说明不会有跨迭代的依赖
    depInfoForLp->setBaseAddrIsCrossIterDep(bd, false);
    if (not depInfoForLp->getIsBaseAddrWrite(bd)) continue;

    for (auto setIter = subAddrs.begin(); setIter != subAddrs.end(); setIter++) {
      for (auto setIter2 = subAddrs.begin(); 1; setIter2++) {
        // 在进行依赖判断的时候，自己和自己也要进行比较，确保在不同的迭代里面他们二者并不相同，如果相同就有可能产生跨循环的依赖
        auto gep1 = *setIter;
        auto gep2 = *setIter2;
        // auto func=gep1->block()->function();
        // func->print(std::cerr);
        auto bd1 = getBaseAddr(gep1, topmana);
        auto bd2 = getBaseAddr(gep2, topmana);
        assert(bd1 == bd);
        assert(bd2 == bd);

        auto gepidx1 = depInfoForLp->getGepIdx(gep1);
        auto gepidx2 = depInfoForLp->getGepIdx(gep2);
        assert(gepidx1->idxList.size() == gepidx2->idxList.size());
        int depType = isTwoGepIdxPossiblySame(gepidx1, gepidx2, lp, defaultIdv);
        if ((depType & dCrossIterTotallyNotSame) != 0) {
          if (setIter2 == setIter) break;
          continue;
        }
        if (((depType & dCrossIterPossiblySame) != 0) or ((depType & dCrossIterTotallySame) != 0)) {
          if (depInfoForLp->getIsSubAddrWrite(*setIter) or
              depInfoForLp->getIsSubAddrWrite(*setIter2)) {
            isParallel = false;
            depInfoForLp->setBaseAddrIsCrossIterDep(bd, true);
          }
        }
        if (setIter2 == setIter) break;
      }
    }
  }
  depInfoForLp->setIsParallel(isParallel);
  // depInfoForLp->print(std::cerr);
}

void DependenceAnalysis::makeGepIdx(ir::Loop* lp, ir::IndVar* idv, GepIdx* gepidx) {
  if (lp == nullptr or idv == nullptr or gepidx == nullptr) {
    std::cerr << "mGI(nullptr) ";
    return;
  }
  std::cerr << "mGI(" << lp->header()->name() << ") ";
  for (auto val : gepidx->idxList) {
    if (gepidx->idxTypes[val] != iELSE) continue;
    if (isSimplyLoopInvariant(lp, val)) gepidx->idxTypes[val] = iLOOPINVARIANT;
    if (idv != nullptr) {
      if (val == idv->phiinst()) gepidx->idxTypes[val] = iIDV;
      if (isIDVPLUSMINUSFORMULA(idv, val, lp)) gepidx->idxTypes[val] = iIDVPLUSMINUSFORMULA;
    }
    if (val->dynCast<ir::CallInst>()) gepidx->idxTypes[val] = iCALL;
    if (val->dynCast<ir::LoadInst>()) gepidx->idxTypes[val] = iLOAD;
  }
}

bool DependenceAnalysis::isSimplyLoopInvariant(ir::Loop* lp, ir::Value* val) {
  std::cerr << "isSLI(" << lp->header()->name() << ") ";
  if (lp == nullptr) {
    std::cerr << "isSLI(lp is nullptr) ";
    return false;
  }
  if (val == nullptr) {
    std::cerr << "isSLI(val is nullptr) ";
    return false;
  }
  if (auto constVal = val->dynCast<ir::ConstantValue>()) {
    return true;  // 常数
  }
  if (auto argVal = val->dynCast<ir::Argument>()) {
    return true;  // 参数
  }
  if (auto instVal = val->dynCast<ir::Instruction>()) {
    return domctx->dominate(instVal->block(), lp->header()) and instVal->block() !=
    lp->header();
  }
  return false;
}

bool DependenceAnalysis::isIDVPLUSMINUSFORMULA(ir::IndVar* idv, ir::Value* val, ir::Loop* lp) {
  if (idv == nullptr or val == nullptr or lp == nullptr) {
    std::cerr << "isIPM(nullptr) ";
    return false;
  }
  std::cerr << "isIA(" << lp->header()->name() << ") ";
  if (auto binaryVal = val->dynCast<ir::BinaryInst>()) {
    auto lval = binaryVal->lValue();
    auto rval = binaryVal->rValue();
    if (binaryVal->valueId() != ir::vADD and binaryVal->valueId() != ir::vSUB) return false;
    bool isLValLpI = isSimplyLoopInvariant(lp, lval);
    bool isRValLpI = isSimplyLoopInvariant(lp, rval);
    if (not isRValLpI and not isLValLpI) return false;
    if (isRValLpI) {
      if (lval == idv->phiinst())
        return true;
      else
        return isIDVPLUSMINUSFORMULA(idv, lval, lp);
    } else {  // isLValLpI
      if (rval == idv->phiinst())
        return true;
      else
        return isIDVPLUSMINUSFORMULA(idv, rval, lp);
    }
  }
  return false;
}

int DependenceAnalysis::isTwoGepIdxPossiblySame(GepIdx* gepidx1,
                                                GepIdx* gepidx2,
                                                ir::Loop* lp,
                                                ir::IndVar* idv) {
  if (gepidx1 == nullptr or gepidx2 == nullptr or lp == nullptr or idv == nullptr) {
    std::cerr << "isGPSame(nullptr) ";
    return 0;
  }
  std::cerr << "isGPSame(" << lp->header()->name() << ") ";
  std::vector<DependenceType> compareAns;
  size_t lim = gepidx1->idxList.size();
  int res = 0;
  for (size_t i = 0; i < lim; i++) {
    auto val1 = gepidx1->idxList[i];
    auto val2 = gepidx2->idxList[i];
    auto type1 = gepidx1->idxTypes[val1];
    auto type2 = gepidx2->idxTypes[val2];
    int outputDepInfo = isTwoIdxPossiblySame(val1, val2, type1, type2, lp, idv);
    res = res | outputDepInfo;
  }
  return res;
}

int DependenceAnalysis::isTwoIdxPossiblySame(ir::Value* val1,
                                             ir::Value* val2,
                                             IdxType type1,
                                             IdxType type2,
                                             ir::Loop* lp,
                                             ir::IndVar* idv) {
  if (val1 == nullptr or val2 == nullptr or lp == nullptr or idv == nullptr) {
    std::cerr << "isIPSame(nullptr) ";
    return 0;
  }
  std::cerr << "isIPSame(" << lp->header()->name() << ") ";
  if (val1 == val2) {  // 自己跟自己进行比较
    switch (type1) {
      case iLOOPINVARIANT:
        return dTotallySame | dCrossIterTotallySame;
        break;
      case iCALL:
        return dTotallySame | dCrossIterPossiblySame;
        break;
      case iIDV:
        return dTotallySame | dCrossIterTotallyNotSame;
        break;
      case iIDVPLUSMINUSFORMULA:
        return dTotallySame | dCrossIterTotallyNotSame;
        break;
      case iINNERIDV:
        return dTotallySame | dCrossIterPossiblySame;
        break;
      case iINNERIDVPLUSMINUSFORMULA:
        return dTotallySame | dCrossIterPossiblySame;
        break;
      case iLOAD:
        return dTotallySame | dCrossIterPossiblySame;  // TODO, 依赖于副作用等分析
        break;
      case iELSE:
        return dPossiblySame | dCrossIterPossiblySame;
        break;
      default:
        break;
    }
  }
  if (type1 == type2) {
    switch (type1) {
      case iCALL: {
        auto callInst1 = val1->dynCast<ir::CallInst>();
        auto callInst2 = val2->dynCast<ir::CallInst>();
        auto callFunc1 = callInst1->callee();
        auto callFunc2 = callInst2->callee();
        if (callFunc1 != callFunc2) return dPossiblySame | dCrossIterPossiblySame;
        // TODO:副作用分析判断这里是纯函数
        if (sectx->isPureFunc(callFunc1))
          return dTotallySame | dCrossIterPossiblySame;
        else
          return dPossiblySame | dCrossIterPossiblySame;
        break;
      }

      case iLOOPINVARIANT: {
        auto constval1 = val1->dynCast<ir::ConstantInteger>();
        auto constval2 = val2->dynCast<ir::ConstantInteger>();
        if (constval1 != nullptr and constval2 != nullptr) {
          if (constval1->i32() == constval2->i32())
            return dTotallySame | dCrossIterTotallySame;
          else
            return dTotallyNotSame | dCrossIterTotallyNotSame;
        } else {
          return dPossiblySame | dCrossIterPossiblySame;
        }
        break;
      }

      case iIDV: {
        // if(val1!=val2){
        //     assert(false and "Error: indvar in a same loop is not same!");
        // }
        return dTotallySame | dCrossIterTotallyNotSame;
        break;
      }

      case iIDVPLUSMINUSFORMULA: {
        std::set<ir::Value*> val1Add;
        std::set<ir::Value*> val1Sub;
        auto curVal1 = val1;
        while (curVal1 != idv->phiinst()) {
          if (auto BInst = curVal1->dynCast<ir::BinaryInst>()) {
            auto lval = BInst->lValue();
            auto rval = BInst->rValue();
            ir::Value* LpIVal;
            bool isLVAL = false;
            if (isSimplyLoopInvariant(lp, lval)) {
              LpIVal = lval;
              curVal1 = rval;
              isLVAL = true;
            } else if (isSimplyLoopInvariant(lp, rval)) {
              LpIVal = rval;
              curVal1 = lval;
            } else {
              std::cerr << "Error:GepIdx is not IDVPLUSMINUSFORMULA!" << std::endl;
              assert(false);
            }
            if (BInst->valueId() == ir::vADD) {
              val1Add.insert(LpIVal);
            } else if (BInst->valueId() == ir::vSUB) {
              if (isLVAL) {
                std::cerr << "Error:GepIdx is a-i formula!" << std::endl;
                assert(false);
              }
              val1Sub.insert(LpIVal);
            }
          } else {
            std::cerr << "this is not a idvplusminus formula!" << std::endl;
            assert(false);
          }
        }
        auto curVal2 = val2;
        while (curVal2 != idv->phiinst()) {
          if (auto BInst = curVal2->dynCast<ir::BinaryInst>()) {
            auto lval = BInst->lValue();
            auto rval = BInst->rValue();
            ir::Value* LpIVal;
            bool isLVAL = false;
            if (isSimplyLoopInvariant(lp, lval)) {
              LpIVal = lval;
              curVal1 = rval;
              isLVAL = true;
            } else if (isSimplyLoopInvariant(lp, rval)) {
              LpIVal = rval;
              curVal1 = lval;
            } else {
              std::cerr << "Error:GepIdx is not IDVPLUSMINUSFORMULA!" << std::endl;
              assert(false);
            }
            if (BInst->valueId() == ir::vADD) {
              if (val1Add.count(LpIVal)) {
                val1Add.erase(LpIVal);
              } else {
                return dPossiblySame | dCrossIterPossiblySame;
              }
            } else if (BInst->valueId() == ir::vSUB) {
              if (isLVAL) {
                std::cerr << "Error:GepIdx is a-i formula!" << std::endl;
                assert(false);
              }
              if (val1Sub.count(LpIVal)) {
                val1Sub.erase(LpIVal);
              } else {
                return dPossiblySame | dCrossIterPossiblySame;
              }
            }
          } else {
            std::cerr << "this is not a idvplusminus formula!" << std::endl;
            assert(false);
          }
        }
        return dTotallySame | dCrossIterTotallySame;

        break;
      }

      case iINNERIDV: {
        return dPossiblySame | dCrossIterPossiblySame;
      }

      case iINNERIDVPLUSMINUSFORMULA: {
        return dPossiblySame | dCrossIterPossiblySame;
      }

      case iLOAD: {
        auto loadInst1 = val1->dynCast<ir::LoadInst>();
        auto loadInst2 = val2->dynCast<ir::LoadInst>();
        auto ptr1 = loadInst1->ptr();
        auto ptr2 = loadInst2->ptr();
        if (ptr1 != ptr2) {
          return dPossiblySame | dCrossIterPossiblySame;
        } else {
          return dPossiblySame | dCrossIterPossiblySame;  // 进行副作用分析可以进一步细化
        }
      }

      case iELSE: {
        return dPossiblySame | dCrossIterPossiblySame;
      }

      default:
        break;
    }
  }
  if ((type1 == iIDV and type2 == iIDVPLUSMINUSFORMULA) or
      (type1 == iIDVPLUSMINUSFORMULA and type2 == iIDV)) {
    return dTotallyNotSame | dCrossIterPossiblySame;
  }
  return dPossiblySame | dCrossIterPossiblySame;
}