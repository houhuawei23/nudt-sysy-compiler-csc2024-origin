#include <iostream>
#include <fstream>
#include <queue>
#include <filesystem>
#include <string_view>
#include "pass/pass.hpp"
#include "pass/analysisinfo.hpp"
#include "pass/analysis/dom.hpp"
#include "mir/mir.hpp"
#include "mir/lowering.hpp"
#include "mir/target.hpp"
#include "mir/iselinfo.hpp"
#include "mir/utils.hpp"
#include "mir/GraphColoringRegisterAllocation.hpp"
#include "mir/fastAllocator.hpp"
#include "mir/FastAllocator.hpp"
#include "mir/linearAllocator.hpp"
#include "target/riscv/RISCVTarget.hpp"
#include "support/StaticReflection.hpp"
#include "support/config.hpp"

namespace fs = std::filesystem;
namespace mir {
/* 保存Runtime相关的Caller-Saved Registers */
void add_external(IPRAUsageCache& infoIPRA) {
  const std::string runtime[] = {"_memset",  "putint",   "getint",    "getch",
                                 "getfloat", "getarray", "getfarray", "putch",
                                 "putarray", "putfloat", "putfarray", "putf"};
  std::unordered_set<RegNum> caller_saved_reg = {
    RISCV::X5,  RISCV::X6,  RISCV::X7,  RISCV::X10, RISCV::X11, RISCV::X12,
    RISCV::X13, RISCV::X14, RISCV::X15, RISCV::X16, RISCV::X17, RISCV::X28,
    RISCV::X29, RISCV::X30, RISCV::X31, RISCV::F10, RISCV::F11, RISCV::F12,
    RISCV::F13, RISCV::F14, RISCV::F15, RISCV::F0,  RISCV::F1,  RISCV::F2,
    RISCV::F3,  RISCV::F4,  RISCV::F5,  RISCV::F6,  RISCV::F7,  RISCV::F28,
    RISCV::F29, RISCV::F30, RISCV::F31, RISCV::F16, RISCV::F17};
  for (auto name : runtime) {
    infoIPRA.add(name, caller_saved_reg);
  }
}

MIROperand FloatPointConstantPool::getFloatConstant(class LoweringContext& ctx,
                                                    float val) {
  uint32_t rep;
  memcpy(&rep, &val, sizeof(float));
  const auto it = mFloatMap.find(rep);
  uint32_t offset;
  if (it != mFloatMap.cend()) {
    offset = it->second;
  } else {
    if (!mFloatDataStorage) {
      auto storage = std::make_unique<MIRDataStorage>(
        MIRDataStorage::Storage{}, true, "float_const_pool", true);
      mFloatDataStorage = storage.get();
      auto pool = std::make_unique<MIRGlobalObject>(
        sizeof(float), std::move(storage), nullptr);
      ctx.module.global_objs().push_back(std::move(pool));
    }
    offset = mFloatMap[rep] =
      mFloatDataStorage->append_word(rep) * sizeof(float);
  }

  const auto ptrType = ctx.getPointerType();
  const auto base = ctx.newVReg(ptrType);
  /* LoadGlobalAddress base, reloc */
  ctx.emitInstBeta(InstLoadGlobalAddress,
                   {base, MIROperand::asReloc(mFloatDataStorage)});

  // Add addr, base, offset
  const auto addr = ctx.newVReg(ptrType);
  ctx.emitInstBeta(InstAdd, {addr, base, MIROperand::asImm(offset, ptrType)});

  // Load dst, addr, 4
  const auto dst = ctx.newVReg(OperandType::Float32);
  ctx.emitInstBeta(InstLoad,
                   {dst, addr, MIROperand::asImm(4, OperandType::Special)});
  return dst;
}

static OperandType get_optype(ir::Type* type) {
  if (type->isInt()) {
    switch (type->btype()) {
      case ir::INT1:
        return OperandType::Bool;
      case ir::INT32:
        return OperandType::Int32;
      default:
        assert(false && "unsupported int type");
    }
  } else if (type->isFloatPoint()) {
    switch (type->btype()) {
      case ir::FLOAT:
        return OperandType::Float32;
      default:
        assert(false && "unsupported float type");
    }
  } else if (type->isPointer()) {
    /* NOTE: rv64 */
    return OperandType::Int64;
  } else {
    return OperandType::Special;
  }
}

MIROperand LoweringContext::newVReg(ir::Type* type) {
  auto optype = get_optype(type);
  return MIROperand::asVReg(codeGenctx->nextId(), optype);
}

MIROperand LoweringContext::newVReg(OperandType type) {
  return MIROperand::asVReg(codeGenctx->nextId(), type);
}

void LoweringContext::addValueMap(ir::Value* ir_val, MIROperand mir_operand) {
  if (valueMap.count(ir_val)) assert(false && "value already mapped");
  valueMap.emplace(ir_val, mir_operand);
}

MIROperand LoweringContext::map2operand(ir::Value* ir_val) {
  assert(ir_val && "null ir_val");
  /* 1. Local Value: alloca */
  if (auto iter = valueMap.find(ir_val); iter != valueMap.end()) {
    return iter->second;
  }

  /* 2. Global Value */
  if (auto gvar = ir_val->dynCast<ir::GlobalVariable>()) {
    auto ptr = newVReg(pointerType);
    /* LoadGlobalAddress ptr, reloc */
    emitInstBeta(InstLoadGlobalAddress,
                 {ptr, MIROperand::asReloc(gvarMap.at(gvar)->reloc.get())});

    return ptr;
  }

  /* 3. Constant */
  if (!ir::isa<ir::Constant>(ir_val)) {
    std::cerr << "error: " << ir_val->name() << " must be constant\n";
    assert(false);
  }

  auto const_val = ir_val->dynCast<ir::Constant>();
  if (const_val->type()->isInt()) {
    auto imm = MIROperand::asImm(const_val->i32(), OperandType::Int32);
    return imm;
  }
  // TODO: support float constant
  if (const_val->type()->isFloat32()) {
    if (auto fpOperand =
          codeGenctx->iselInfo->materializeFPConstant(const_val->f32(), *this);
        fpOperand.isInit()) {
      return fpOperand;
    }

    auto fpOperand =
      mFloatConstantPool.getFloatConstant(*this, const_val->f32());

    return fpOperand;
  }
  std::cerr << "Map2Operand Error: Not Supported IR Value Type: "
            << utils::enumName(static_cast<ir::BType>(ir_val->type()->btype()))
            << std::endl;
  assert(false && "Not Supported Type.");
  return MIROperand{};
}

void LoweringContext::emitCopy(MIROperand dst, MIROperand src) {
  /* copy dst, src */
  emitInstBeta(select_copy_opcode(dst, src), {dst, src});
}

void createMIRModule(ir::Module& ir_module,
                     MIRModule& mir_module,
                     Target& target,
                     pass::TopAnalysisInfoManager* tAIM);

void createMIRFunction(ir::Function* ir_func,
                       MIRFunction* mir_func,
                       CodeGenContext& codegen_ctx,
                       LoweringContext& lowering_ctx,
                       pass::TopAnalysisInfoManager* tAIM);

void createMIRInst(ir::Instruction* ir_inst, LoweringContext& ctx);

void lower_GetElementPtr_beta(ir::inst_iterator begin,
                              ir::inst_iterator end,
                              LoweringContext& ctx);

std::unique_ptr<MIRModule> createMIRModule(ir::Module& ir_module,
                                           Target& target,
                                           pass::TopAnalysisInfoManager* tAIM) {
  auto mir_module_uptr = std::make_unique<MIRModule>(&ir_module, target);
  createMIRModule(ir_module, *mir_module_uptr, target, tAIM);
  return mir_module_uptr;
}
void createMIRModuleBeta(ir::Module& ir_module,
                         MIRModule& mir_module,
                         Target& target,
                         pass::TopAnalysisInfoManager* tAIM) {
  createMIRModule(ir_module, mir_module, target, tAIM);
}

void createMIRModule(ir::Module& ir_module,
                     MIRModule& mir_module,
                     Target& target,
                     pass::TopAnalysisInfoManager* tAIM) {
  auto& config = sysy::Config::getInstance();

  bool debugLowering = config.logLevel >= sysy::LogLevel::DEBUG;

  auto& functions = mir_module.functions();      // uptr vector
  auto& global_objs = mir_module.global_objs();  // uptr vector

  LoweringContext lowering_ctx(mir_module, target);
  auto& func_map = lowering_ctx.funcMap;  // ir func -> mir func
  auto& gvar_map = lowering_ctx.gvarMap;  // ir gvar -> mir gobj

  //! 1. for all functions, create MIRFunction
  for (auto func : ir_module.funcs()) {
    functions.push_back(
      std::make_unique<MIRFunction>(func->name(), &mir_module));
    func_map.emplace(func, functions.back().get());
  }

  //! 2. for all global variables, create MIRGlobalObject
  for (auto ir_gvar : ir_module.globalVars()) {
    constexpr bool DebugGlobal = false;
    const auto name = ir_gvar->name().substr(1); /* remove '@' */
    /* 基础类型 (int OR float) */
    auto type = ir_gvar->type()->dynCast<ir::PointerType>()->baseType();
    if (type->isArray()) type = dyn_cast<ir::ArrayType>(type)->baseType();
    const size_t size = type->size();
    const bool read_only = ir_gvar->isConst();
    const bool is_float = type->isFloat32();
    const size_t align = 4;

    if (ir_gvar->isInit()) { /* .data: 已初始化的、可修改的全局数据 (Array and
                                Scalar) */
      /* NOTE: 全局变量初始化一定为常值表达式 */
      if (DebugGlobal)
        std::cerr << "init size is " << ir_gvar->init_cnt() << "\n";
      MIRDataStorage::Storage data;
      for (int i = 0; i < ir_gvar->init_cnt(); i++) {
        const auto constValue = dyn_cast<ir::Constant>(ir_gvar->init(i));
        /* NOTE: float to uint32_t, type cast, doesn't change the memory */
        uint32_t word;
        if (type->isInt()) {
          const auto val = constValue->i32();
          memcpy(&word, &val, sizeof(uint32_t));
        } else if (type->isFloat32()) {
          const auto val = constValue->f32();
          memcpy(&word, &val, sizeof(float));
        } else {
          assert(false && "Not Supported Type.");
        }
        data.push_back(word);
      }
      auto mir_storage = std::make_unique<MIRDataStorage>(
        std::move(data), read_only, name, is_float);
      auto mir_gobj = std::make_unique<MIRGlobalObject>(
        align, std::move(mir_storage), &mir_module);
      mir_module.global_objs().push_back(std::move(mir_gobj));
    } else { /* .bss: 未初始化的全局数据 (Just Scalar) */
      auto mir_storage = std::make_unique<MIRZeroStorage>(size, name, is_float);
      auto mir_gobj = std::make_unique<MIRGlobalObject>(
        align, std::move(mir_storage), &mir_module);
      mir_module.global_objs().push_back(std::move(mir_gobj));
    }
    gvar_map.emplace(ir_gvar, mir_module.global_objs().back().get());
  }

  // TODO: transformModuleBeforeCodeGen

  //! 3. codegen
  CodeGenContext codegen_ctx{target, target.getDataLayout(),
                             target.getTargetInstInfo(),
                             target.getTargetFrameInfo(), MIRFlags{}};
  codegen_ctx.iselInfo = &target.getTargetIselInfo();
  codegen_ctx.scheduleModel = &target.getScheduleModel();
  lowering_ctx.codeGenctx = &codegen_ctx;

  /* 缓存各个函数所用到的Caller-Saved Registers */
  IPRAUsageCache infoIPRA;

  auto dumpStageWithMsg = [&](std::ostream& os, std::string_view stage,
                              std::string_view msg) {
    enum class Style { RED, BOLD, RESET };

    static std::unordered_map<Style, std::string_view> styleMap = {
      {Style::RED, "\033[0;31m"},
      {Style::BOLD, "\033[1m"},
      {Style::RESET, "\033[0m"}};

    os << "\n";
    os << styleMap[Style::RED] << styleMap[Style::BOLD];
    os << "[" << stage << "] ";
    os << styleMap[Style::RESET];
    os << msg << std::endl;
  };

  //! 4. Lower all Functions
  add_external(infoIPRA);
  for (auto& ir_func : ir_module.funcs()) {
    if (ir_func->blocks().empty()) continue;
    /* Just for Debug */
    size_t stageIdx = 0;
    auto dumpStageResult = [&](std::string stage, MIRFunction* mir_func,
                               CodeGenContext& codegen_ctx) {
      if (!debugLowering) return;
      auto fileName =
        mir_func->name() + std::to_string(stageIdx) + "_" + stage + ".ll";
      auto path = config.debugDir() / fs::path(fileName);
      std::ofstream fout(path);
      mir_func->print(fout, codegen_ctx);
      stageIdx++;
    };
    if (debugLowering) {
      auto fileName =
        ir_func->name() + std::to_string(stageIdx) + "_" + "BeforeLowering.ll";
      auto path = config.debugDir() / fs::path(fileName);
      std::ofstream fout(path);
      ir_func->print(fout);
      stageIdx++;
    }

    const auto mir_func = func_map[ir_func];
    /* lower function body to generic MIR */
    {
      createMIRFunction(ir_func, mir_func, codegen_ctx, lowering_ctx, tAIM);
      dumpStageResult("AfterLowering", mir_func, codegen_ctx);
      if (!mir_func->verify(std::cerr, codegen_ctx)) {
        std::cerr << "Lowering Error: " << mir_func->name()
                  << " failed to verify.\n";
      }
    }

    /* instruction selection */
    {
      ISelContext isel_ctx(codegen_ctx);
      isel_ctx.runInstSelect(mir_func);
      dumpStageResult("AfterIsel", mir_func, codegen_ctx);
    }
    /* Optimize: register coalescing */

    /* Optimize: peephole optimization (窥孔优化) */
    {
      while (genericPeepholeOpt(*mir_func, codegen_ctx))
        ;
      dumpStageResult("AfterPeephole", mir_func, codegen_ctx);
    }

    /* pre-RA legalization */

    /* Optimize: pre-RA scheduling, minimize register usage */
    {
      // preRASchedule(*mir_func, codegen_ctx);
      // dumpStageResult("AfterPreRASchedule", mir_func, codegen_ctx);
    }

    /* register allocation */
    {
      codegen_ctx.registerInfo = new RISCVRegisterInfo();
      if (codegen_ctx.registerInfo) {
        // GraphColoringAllocate(*mir_func, codegen_ctx, infoIPRA);
        // fastAllocator(*mir_func, codegen_ctx, infoIPRA);
        fastAllocatorBeta(*mir_func, codegen_ctx, infoIPRA);
        dumpStageResult("AfterGraphColoring", mir_func, codegen_ctx);
      }
    }

    /* stack allocation */
    if (codegen_ctx.registerInfo) {
      /* after sa, all stack objects are allocated with .offset */
      allocateStackObjects(mir_func, codegen_ctx);
      codegen_ctx.flags.postSA = true;
      dumpStageResult("AfterStackAlloc", mir_func, codegen_ctx);
    }

    // {
    //     /* post-RA scheduling, minimize cycles */
    //     postRASchedule(*mir_func, codegen_ctx);
    //     dumpStageResult("AfterPostRASchedule", mir_func, codegen_ctx);
    // }

    /* post legalization */
    postLegalizeFunc(*mir_func, codegen_ctx);

    /* Add Function to IPRA cache */
    if (codegen_ctx.registerInfo) {
      infoIPRA.add(codegen_ctx, *mir_func);
    }

    dumpStageResult("AfterCodeGen", mir_func, codegen_ctx);

    if (not target.verify(*mir_func)) {
      std::cerr << "Lowering Error: " << mir_func->name()
                << " failed to verify." << std::endl;
    }
  }  // end of for (auto& ir_func : ir_module.funcs())
  /* module verify */
}

void createMIRFunction(ir::Function* ir_func,
                       MIRFunction* mir_func,
                       CodeGenContext& codegen_ctx,
                       LoweringContext& lowering_ctx,
                       pass::TopAnalysisInfoManager* tAIM) {
  if (ir_func->blocks().empty()) return;
  const auto& config = sysy::Config::getInstance();
  lowering_ctx.setCurrFunc(mir_func);
  /* Some Debug Information */
  constexpr bool DebugCreateMirFunction = false;

  // TODO: before lowering, get some analysis pass result

  auto domCtx = tAIM->getDomTree(ir_func);
  domCtx->refresh();
  domCtx->BFSDomTreeInfoRefresh();
  auto irBlocks = domCtx->BFSDomTreeVector();

  // std::cerr << "000" << std::endl;
  // for (auto block : ir_func->blocks()) {
  //   std::cerr << "block: " << block->name()
  //             << ", size: " << block->insts().size() << "addr: " << block
  //             << std::endl;
  // }

  // std::cerr << "111" << std::endl;
  // for (auto block : irBlocks) {
  //   std::cerr << "block: " << block->name()
  //             << ", size: " << block->insts().size() << "addr: " << block
  //             << std::endl;
  // }
  // std::cerr << "222" << std::endl;

  for (auto block : ir_func->blocks()) {
    if (std::find(irBlocks.begin(), irBlocks.end(), block) == irBlocks.end()) {
      // std::cerr << "Some Blocks are not in dom tree" << std::endl;
      // std::cerr << block->name() << "size: " << block->insts().size()
      //           << "addr: " << block << std::endl;
      irBlocks.push_back(block);
    }
  }
  // std::cerr << "333" << std::endl;
  // std::cerr << "irBlocks size: " << irBlocks.size() << std::endl;
  // for (auto block : irBlocks) {
  //   std::cerr << "block: " << block->name()
  //             << ", size: " << block->insts().size() << "addr: " << block
  //             << std::endl;
  // }

  //! 1. map from ir to mir
  auto& block_map = lowering_ctx.blockMap;
  auto& target = codegen_ctx.target;
  auto& datalayout = target.getDataLayout();

  for (auto ir_block : irBlocks) {
    mir_func->blocks().push_back(std::make_unique<MIRBlock>(
      mir_func, "label" + std::to_string(codegen_ctx.nextLabelId())));
    block_map.emplace(ir_block, mir_func->blocks().back().get());
  }

  //! 2. emitPrologue for function
  {
    /* assign vreg to arg */
    for (auto ir_arg : ir_func->args()) {
      auto vreg = lowering_ctx.newVReg(ir_arg->type());
      lowering_ctx.addValueMap(ir_arg, vreg);
      mir_func->args().push_back(vreg);
    }
    lowering_ctx.setCurrBlock(block_map.at(ir_func->entry()));
    codegen_ctx.frameInfo.emitPrologue(mir_func, lowering_ctx);
  }
  if (DebugCreateMirFunction)
    std::cerr << "stage 2: emitPrologue for function" << std::endl;

  //! 3. process alloca, new stack object for each alloca
  lowering_ctx.setCurrBlock(block_map.at(ir_func->entry()));  // entry
  for (auto& ir_inst : ir_func->entry()->insts()) {
    // NOTE: all alloca in entry
    if (not ir_inst->isa<ir::AllocaInst>()) continue;

    const auto ir_alloca = dyn_cast<ir::AllocaInst>(ir_inst);

    auto pointee_type = ir_alloca->baseType();
    // std::cerr << ir_alloca->name() << std::endl;
    // std::cerr << "type: " << *pointee_type << std::endl;
    // std::cerr << "size: " << pointee_type->size() << std::endl;

    uint32_t align = 4;  // TODO: align, need bind to ir object
    auto storage = mir_func->newStackObject(
      codegen_ctx.nextId(),                         // id
      static_cast<uint32_t>(pointee_type->size()),  // size
      align,                                        // align
      0,                                            // offset
      StackObjectUsage::Local);
    // emit load stack object addr inst
    auto addr = lowering_ctx.newVReg(lowering_ctx.getPointerType());

    lowering_ctx.emitInstBeta(InstLoadStackObjectAddr, {addr, storage});
    // map
    lowering_ctx.addValueMap(ir_inst, addr);
  }

  //! 4. lowering all blocks
  {
    for (auto& ir_block : irBlocks) {
      auto mir_block = block_map[ir_block];
      lowering_ctx.setCurrBlock(mir_block);

      auto& insts = ir_block->insts();
      for (auto iter = insts.begin(); iter != insts.end();) {
        auto ir_inst = *iter;
        if (ir_inst->isa<ir::AllocaInst>()) {
          iter++;
          continue;
        } else if (ir_inst->isa<ir::GetElementPtrInst>()) {
          auto ir_getelement_inst = dyn_cast<ir::GetElementPtrInst>(ir_inst);
          int id = ir_getelement_inst->getid();
          if (id == 0) {
            createMIRInst(ir_inst, lowering_ctx);
            iter++;
          } else {
            auto end = iter;
            end++;
            while (end != insts.end() && (*end)->isa<ir::GetElementPtrInst>()) {
              auto preInst = std::prev(end);
              auto endInst = dyn_cast<ir::GetElementPtrInst>(*end);
              if (endInst->value() == (*preInst) && endInst->getid() != 0) {
                end++;
              } else {
                break;
              }
            }
            lower_GetElementPtr_beta(iter, end, lowering_ctx);  // [iter, end)
            iter = end;
          }
        } else {
          createMIRInst(ir_inst, lowering_ctx);
          iter++;
        }
        if (DebugCreateMirFunction) {
          ir_inst->print(std::cerr);
          std::cerr << std::endl;
        }
      }
    }
  }
  if (DebugCreateMirFunction) {
    std::cerr << "stage 4: lowering all blocks" << std::endl;
  }
}

void lower(ir::UnaryInst* ir_inst, LoweringContext& ctx);
void lower(ir::BinaryInst* ir_inst, LoweringContext& ctx);
void lower(ir::BranchInst* ir_inst, LoweringContext& ctx);
void lower(ir::LoadInst* ir_inst, LoweringContext& ctx);
void lower(ir::StoreInst* ir_inst, LoweringContext& ctx);
void lower(ir::ICmpInst* ir_inst, LoweringContext& ctx);
void lower(ir::FCmpInst* ir_inst, LoweringContext& ctx);
void lower(ir::CallInst* ir_inst, LoweringContext& ctx);
void lower(ir::ReturnInst* ir_inst, LoweringContext& ctx);
void lower(ir::BranchInst* ir_inst, LoweringContext& ctx);
void lower(ir::BitCastInst* ir_inst, LoweringContext& ctx);
void lower(ir::MemsetInst* ir_inst, LoweringContext& ctx);
void lower(ir::GetElementPtrInst* ir_inst, LoweringContext& ctx);

void createMIRInst(ir::Instruction* ir_inst, LoweringContext& ctx) {
  switch (ir_inst->valueId()) {
    case ir::ValueId::vFNEG:
    case ir::ValueId::vTRUNC:
    case ir::ValueId::vZEXT:
    case ir::ValueId::vSEXT:
    case ir::ValueId::vFPTRUNC:
    case ir::ValueId::vFPTOSI:
    case ir::ValueId::vSITOFP:
      lower(dyn_cast<ir::UnaryInst>(ir_inst), ctx);
      break;
    case ir::ValueId::vADD:
    case ir::ValueId::vFADD:
    case ir::ValueId::vSUB:
    case ir::ValueId::vFSUB:
    case ir::ValueId::vMUL:
    case ir::ValueId::vFMUL:
    case ir::ValueId::vUDIV:
    case ir::ValueId::vSDIV:
    case ir::ValueId::vFDIV:
    case ir::ValueId::vUREM:
    case ir::ValueId::vSREM:
    case ir::ValueId::vFREM:
      lower(dyn_cast<ir::BinaryInst>(ir_inst), ctx);
      break;
    case ir::ValueId::vIEQ:
    case ir::ValueId::vINE:
    case ir::ValueId::vISGT:
    case ir::ValueId::vISGE:
    case ir::ValueId::vISLT:
    case ir::ValueId::vISLE:
      lower(dyn_cast<ir::ICmpInst>(ir_inst), ctx);
      break;
    case ir::ValueId::vFOEQ:
    case ir::ValueId::vFONE:
    case ir::ValueId::vFOGT:
    case ir::ValueId::vFOGE:
    case ir::ValueId::vFOLT:
    case ir::ValueId::vFOLE:
      lower(dyn_cast<ir::FCmpInst>(ir_inst), ctx);
      break;
    case ir::ValueId::vALLOCA:
      std::cerr << "alloca not supported" << std::endl;
      break;
    case ir::ValueId::vLOAD:
      lower(dyn_cast<ir::LoadInst>(ir_inst), ctx);
      break;
    case ir::ValueId::vSTORE:
      lower(dyn_cast<ir::StoreInst>(ir_inst), ctx);
      break;
    case ir::ValueId::vGETELEMENTPTR:
      lower(dyn_cast<ir::GetElementPtrInst>(ir_inst), ctx);
      break;
    case ir::ValueId::vRETURN:
      lower(dyn_cast<ir::ReturnInst>(ir_inst), ctx);
      break;
    case ir::ValueId::vBR:
      lower(dyn_cast<ir::BranchInst>(ir_inst), ctx);
      break;
    case ir::ValueId::vCALL:
      lower(dyn_cast<ir::CallInst>(ir_inst), ctx);
      break;
    case ir::ValueId::vMEMSET:
      lower(dyn_cast<ir::MemsetInst>(ir_inst), ctx);
      break;
    case ir::ValueId::vBITCAST:
      lower(dyn_cast<ir::BitCastInst>(ir_inst), ctx);
      break;
    default:
      const auto valueIdEnumName =
        utils::enumName(static_cast<ir::ValueId>(ir_inst->valueId()));
      std::cerr << valueIdEnumName << ": not supported inst" << std::endl;
      assert(false && "not supported inst");
  }
}
void lower(ir::UnaryInst* ir_inst, LoweringContext& ctx) {
  auto gc_instid = [scid = ir_inst->valueId()] {
    switch (scid) {
      case ir::ValueId::vFNEG:
        return InstFNeg;
      case ir::ValueId::vTRUNC:
        return InstTrunc;
      case ir::ValueId::vZEXT:
        return InstZExt;
      case ir::ValueId::vSEXT:
        return InstSExt;
      case ir::ValueId::vFPTRUNC:
        assert(false && "not supported unary inst");
      case ir::ValueId::vFPTOSI:
        return InstF2S;
      case ir::ValueId::vSITOFP:
        return InstS2F;
      default:
        assert(false && "not supported unary inst");
    }
  }();

  /**
   * IR: ir_inst := value [valueId]
   * -> MIR: InstXXX dst, src */
  auto dst = ctx.newVReg(ir_inst->type());
  ctx.emitInstBeta(gc_instid, {dst, ctx.map2operand(ir_inst->value())});
  ctx.addValueMap(ir_inst, dst);
}

/*
 * @brief: Lowering ICmpInst (int OR float)
 * @note:
 * 1. int
 *   IR: <result> = icmp <cond> <ty> <op1>, <op2>
 *   MIRGeneric: ICmp dst, src1, src2, op
 * 2. float
 *   IR: <result> = fcmp [fast-math flags]* <cond> <ty> <op1>, <op2>
 *   MIRGeneric: FCmp dst, src1, src2, op
 */
void lower(ir::ICmpInst* ir_inst, LoweringContext& ctx) {
  auto op = [scid = ir_inst->valueId()] {
    switch (scid) {
      case ir::ValueId::vIEQ:
        return CompareOp::ICmpEqual;
      case ir::ValueId::vINE:
        return CompareOp::ICmpNotEqual;
      case ir::ValueId::vISGT:
        return CompareOp::ICmpSignedGreaterThan;
      case ir::ValueId::vISGE:
        return CompareOp::ICmpSignedGreaterEqual;
      case ir::ValueId::vISLT:
        return CompareOp::ICmpSignedLessThan;
      case ir::ValueId::vISLE:
        return CompareOp::ICmpSignedLessEqual;
      default:
        assert(false && "not supported icmp inst");
    }
  }();

  auto dst = ctx.newVReg(ir_inst->type());

  ctx.emitInstBeta(
    InstICmp,
    {dst, ctx.map2operand(ir_inst->lhs()), ctx.map2operand(ir_inst->rhs()),
     MIROperand::asImm(static_cast<uint32_t>(op), OperandType::Special)});

  ctx.addValueMap(ir_inst, dst);
}
void lower(ir::FCmpInst* ir_inst, LoweringContext& ctx) {
  auto op = [scid = ir_inst->valueId()] {
    switch (scid) {
      case ir::ValueId::vFOEQ:
        return CompareOp::FCmpOrderedEqual;
      case ir::ValueId::vFONE:
        return CompareOp::FCmpOrderedNotEqual;
      case ir::ValueId::vFOGT:
        return CompareOp::FCmpOrderedGreaterThan;
      case ir::ValueId::vFOGE:
        return CompareOp::FCmpOrderedGreaterEqual;
      case ir::ValueId::vFOLT:
        return CompareOp::FCmpOrderedLessThan;
      case ir::ValueId::vFOLE:
        return CompareOp::FCmpOrderedLessEqual;
      default:
        assert(false && "not supported fcmp inst");
    }
  }();

  auto dst = ctx.newVReg(ir_inst->type());
  ctx.emitInstBeta(
    InstFCmp,
    {dst, ctx.map2operand(ir_inst->lhs()), ctx.map2operand(ir_inst->rhs()),
     MIROperand::asImm(static_cast<uint32_t>(op), OperandType::Special)});
  ctx.addValueMap(ir_inst, dst);
}

/* CallInst */
void lower(ir::CallInst* ir_inst, LoweringContext& ctx) {
  ctx.mTarget.getTargetFrameInfo().emitCall(ir_inst, ctx);
}

/* BinaryInst */
void lower(ir::BinaryInst* ir_inst, LoweringContext& ctx) {
  auto gc_instid = [scid = ir_inst->valueId()] {
    switch (scid) {
      case ir::ValueId::vADD:
        return InstAdd;
      case ir::ValueId::vFADD:
        return InstFAdd;
      case ir::ValueId::vSUB:
        return InstSub;
      case ir::ValueId::vFSUB:
        return InstFSub;
      case ir::ValueId::vMUL:
        return InstMul;
      case ir::ValueId::vFMUL:
        return InstFMul;
      case ir::ValueId::vUDIV:
        return InstUDiv;
      case ir::ValueId::vSDIV:
        return InstSDiv;
      case ir::ValueId::vFDIV:
        return InstFDiv;
      case ir::ValueId::vUREM:
        return InstURem;
      case ir::ValueId::vSREM:
        return InstSRem;
      default:
        assert(false && "not supported binary inst");
    }
  }();
  /** IR: ir_inst := lValue, rValue [ValueId]
   * -> MIR: InstXXX dst, src1, src2
   */
  auto dst = ctx.newVReg(ir_inst->type());

  ctx.emitInstBeta(gc_instid, {dst, ctx.map2operand(ir_inst->lValue()),
                               ctx.map2operand(ir_inst->rValue())});
  ctx.addValueMap(ir_inst, dst);
}

/* BranchInst */
void emit_branch(ir::BasicBlock* srcblock,
                 ir::BasicBlock* dstblock,
                 LoweringContext& lctx);

void lower(ir::BranchInst* ir_inst, LoweringContext& ctx) {
  auto src_block = ir_inst->block();
  auto mir_func = ctx.currFunc();
  const auto codegen_ctx = ctx.codeGenctx;
  if (ir_inst->is_cond()) {  // conditional branch
    /*
        branch cond, iftrue, iffalse
        -> MIR
    preblock:
        ...
        branch cond, iftrue

    nextblock:
        jump iffalse

        ...
    */
    /* branch cond, iftrue */
    ctx.emitInstBeta(
      InstBranch,
      {
        ctx.map2operand(ir_inst->cond()) /* cond */,
        MIROperand::asReloc(ctx.map2block(ir_inst->iftrue())) /* iftrue */,
        MIROperand::asProb(0.5) /* prob*/
      });

    /* nextblock: jump iffalse */
    auto findBlockIter = [mir_func](const MIRBlock* block) {
      return std::find_if(mir_func->blocks().begin(), mir_func->blocks().end(),
                          [block](const std::unique_ptr<MIRBlock>& mir_block) {
                            return mir_block.get() == block;
                          });
    };
    {
      /* insert new block after current block */
      auto curBlockIter = findBlockIter(ctx.currBlock());
      assert(curBlockIter != mir_func->blocks().end());

      auto newBlock = std::make_unique<MIRBlock>(
        ctx.currFunc(), "label" + std::to_string(codegen_ctx->nextLabelId()));
      auto newBlockPtr = newBlock.get();
      // insert new block after current block
      mir_func->blocks().insert(++curBlockIter, std::move(newBlock));
      ctx.setCurrBlock(newBlockPtr);
    }
    /* emit jump to iffalse */
    ctx.emitInstBeta(InstJump,
                     {MIROperand::asReloc(ctx.map2block(ir_inst->iffalse()))});
  } else {  // unconditional branch
    auto dst_block = ir_inst->dest();
    emit_branch(src_block, dst_block, ctx);
  }
}
void emit_branch(ir::BasicBlock* srcblock,
                 ir::BasicBlock* dstblock,
                 LoweringContext& lctx) {
  lctx.emitInstBeta(InstJump, {MIROperand::asReloc(lctx.map2block(dstblock))});
}

/** LoadInst
 * IR: inst := ptr [ValueId: vLOAD]
 * -> MIR: InstLoad dst, src, align
 */
void lower(ir::LoadInst* ir_inst, LoweringContext& ctx) {
  const uint32_t align = 4;

  auto inst = ctx.emitInstBeta(
    InstLoad, {
                ctx.newVReg(ir_inst->type()),                    // dst
                ctx.map2operand(ir_inst->ptr()),                 // src
                MIROperand::asImm(align, OperandType::Alignment) /* align*/
              });

  ctx.addValueMap(ir_inst, inst->operand(0));
}

/** StoreInst
 * IR: inst := value, ptr [ValueId: vSTORE]
 * -> MIR: InstStore addr, src, align
 */
void lower(ir::StoreInst* ir_inst, LoweringContext& ctx) {
  ctx.emitInstBeta(InstStore,
                   {
                     ctx.map2operand(ir_inst->ptr()),              // addr
                     ctx.map2operand(ir_inst->value()),            // src
                     MIROperand::asImm(4, OperandType::Alignment)  // align
                   });
}

/* ReturnInst */
void lower(ir::ReturnInst* ir_inst, LoweringContext& ctx) {
  ctx.mTarget.getTargetFrameInfo().emitReturn(ir_inst, ctx);
}

/* BitCastInst */
void lower(ir::BitCastInst* ir_inst, LoweringContext& ctx) {
  const auto base = ir_inst->value();
  ctx.addValueMap(ir_inst, ctx.map2operand(base));
}

/* MemsetInst */
void lower(ir::MemsetInst* ir_inst, LoweringContext& ctx) {
  const auto ir_pointer = ir_inst->value();
  const auto size =
    dyn_cast<ir::PointerType>(ir_pointer->type())->baseType()->size();

  /* 通过寄存器传递参数 */
  // 1. 指针
  {
    auto val = ctx.map2operand(ir_pointer);
    auto dst = MIROperand::asISAReg(RISCV::X10, OperandType::Int64);
    // assert(dst);
    ctx.emitCopy(dst, val);
  }

  // 2. 长度
  {
    auto val = ctx.map2operand(ir::Constant::gen_i32(size));
    auto dst = MIROperand::asISAReg(RISCV::X11, OperandType::Int64);
    // assert(dst);
    ctx.emitCopy(dst, val);
  }

  /* 生成跳转至被调用函数的指令 */
  ctx.emitInstBeta(RISCV::JAL, {MIROperand::asReloc(ctx.memsetFunc)});
}

/* GetElementPtrInst */
/*
 * @brief: lower GetElementPtrInst for Pointer
 * @note:
 *    Pointer: <result> = getelementptr <type>, <type>* <ptrval>, i32 <idx>
 * @details:
 *    1. 生成add AND mul指令来计算偏移量
 *    2. 生成add指令来计算得到目标指针地址
 */
void lower(ir::GetElementPtrInst* ir_inst, LoweringContext& ctx) {
  constexpr bool Debug = false;
  if (Debug) {
    std::cerr << "lower GetElementPtrInst for Pointer. \n";
    ir_inst->print(std::cerr);
    std::cerr << "\n";
  }

  auto base = ctx.map2operand(ir_inst->value());  // 基地址
  MIROperand ptr = base;
  auto btype = ir_inst->baseType();
  int stride = 1;
  if (btype->isArray()) {
    auto dims = dyn_cast<ir::ArrayType>(btype)->dims();
    for (int i = 0; i < dims.size(); i++) {
      stride *= dims[i];
    }
    // stride = dyn_cast<ir::ArrayType>(btype)->dims()[0];
  }
  auto ir_index = ir_inst->index();

  if (auto ir_constant = dyn_cast<ir::Constant>(ir_index)) {
    auto newPtr = ctx.newVReg(OperandType::Int64);
    ctx.emitInstBeta(
      InstAdd,
      {newPtr, ptr,
       MIROperand::asImm(4 * stride * ir_constant->i32(), OperandType::Int64)});
    ptr = newPtr;
  } else {
    auto newPtr_mul = ctx.newVReg(OperandType::Int64);
    ctx.emitInstBeta(InstMul,
                     {newPtr_mul, ctx.map2operand(ir_index),
                      MIROperand::asImm(4 * stride, OperandType::Int64)});
    auto newPtr_add = ctx.newVReg(OperandType::Int64);
    ctx.emitInstBeta(InstAdd, {newPtr_add, ptr, newPtr_mul});
    ptr = newPtr_add;
  }

  ctx.addValueMap(ir_inst, ptr);
}
/*
 * @brief: lower GetElementPtrInst [begin, end) for Array
 * @note:
 *      1. Array: <result> = getelementptr <type>, <type>* <ptrval>, i32 0, i32
 * <idx>
 * @details:
 *    How to compute?
 *      我们遍历每一维度的下标索引, 直接在当前维度计算出当前维度的偏移量
 */
void lower_GetElementPtr_beta(ir::inst_iterator begin,
                              ir::inst_iterator end,
                              LoweringContext& ctx) {
  constexpr bool Debug = false;
  if (Debug) {
    std::cerr << "lower GetElementPtrInst for Array. \n";
    auto iter = begin;
    while (iter != end) {
      auto ir_inst = dyn_cast<ir::GetElementPtrInst>(*iter);

      /* Instruction */
      std::cerr << "the instruction is: ";
      ir_inst->print(std::cerr);
      std::cerr << "\n";

      /* Attribute */
      std::cerr << "the attribute: \n";
      std::cerr << "\tindex is: " << ir_inst->index()->name() << "\n";
      std::cerr << "\tid is: " << ir_inst->getid() << "\n";

      std::cerr << "\n";
      iter++;
    }
  }

  auto base = ctx.map2operand(
    dyn_cast<ir::GetElementPtrInst>(*begin)->value());  // 基地址
  auto iter = begin;
  MIROperand ptr = base;

  auto ir_inst = dyn_cast<ir::GetElementPtrInst>(*iter);
  ir::Value* instEnd = *iter;  // GetElementPtr指令末尾 (包含)

  MIROperand mir_offset = ctx.map2operand(ir_inst->index());
  auto dims = ir_inst->cur_dims();
  /* 乘法运算 */
  for (int i = 1; i < dims.size(); i++) {
    auto newPtr = ctx.newVReg(OperandType::Int64);
    ctx.emitInstBeta(InstMul, {newPtr, mir_offset,
                               MIROperand::asImm(dims[i], OperandType::Int64)});
    mir_offset = newPtr;
  }
  /* 1. 偏移量 */
  {
    auto newPtr = ctx.newVReg(OperandType::Int64);
    ctx.emitInstBeta(
      InstMul, {newPtr, mir_offset, MIROperand::asImm(4, OperandType::Int64)});
    mir_offset = newPtr;
  }

  /* 2. 指针运算 */
  {
    auto newPtr = ctx.newVReg(OperandType::Int64);
    ctx.emitInstBeta(InstAdd, {newPtr, ptr, mir_offset});
    ptr = newPtr;
    ctx.addValueMap(instEnd, ptr);
  }
  iter++;

  while (iter != end) {
    ir_inst = dyn_cast<ir::GetElementPtrInst>(*iter);
    dims = ir_inst->cur_dims();
    instEnd = *iter;
    mir_offset = ctx.map2operand(ir_inst->index());

    /* 乘法运算 */
    for (int i = 1; i < dims.size(); i++) {
      auto newPtr = ctx.newVReg(OperandType::Int64);
      ctx.emitInstBeta(
        InstMul,
        {newPtr, mir_offset, MIROperand::asImm(dims[i], OperandType::Int64)});
      mir_offset = newPtr;
    }

    /* 偏移量 */
    {
      auto newPtr = ctx.newVReg(OperandType::Int64);
      ctx.emitInstBeta(InstMul, {newPtr, mir_offset,
                                 MIROperand::asImm(4, OperandType::Int64)});
      mir_offset = newPtr;
    }

    /* 指针运算 */
    {
      auto newPtr = ctx.newVReg(OperandType::Int64);
      ctx.emitInstBeta(InstAdd, {newPtr, ptr, mir_offset});
      ptr = newPtr;
      ctx.addValueMap(instEnd, ptr);
    }

    iter++;
  }
}
}  // namespace mir