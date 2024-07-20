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
#include "mir/linearAllocator.hpp"
#include "target/riscv/riscvtarget.hpp"
#include "support/StaticReflection.hpp"
#include <iostream>
#include <fstream>
#include <queue>

namespace mir {


MIROperand* FloatPointConstantPool::getFloatConstant(class LoweringContext& ctx, float val){
  uint32_t rep;
  memcpy(&rep, &val, sizeof(float));
  const auto it = mFloatMap.find(rep);
  uint32_t offset;
  if(it != mFloatMap.cend()) {
    offset = it->second;
  } else {
    if (not mFloatDataStorage) {
        auto storage = std::make_unique<MIRDataStorage>(MIRDataStorage::Storage{}, true, "float_const_pool", true);
        mFloatDataStorage = storage.get();
        auto pool = std::make_unique<MIRGlobalObject>(sizeof(float), std::move(storage), nullptr);
        ctx.get_mir_module().global_objs().push_back(std::move(pool));
    }
    offset = mFloatMap[rep] = mFloatDataStorage->append_word(rep) * sizeof(float);
  }

  const auto ptrType = ctx.get_ptr_type();
  const auto base = ctx.new_vreg(ptrType);

  // LoadGlobalAddress base, reloc(mFloatDataStorage)
  auto loadGlobalAddrInst = new MIRInst(InstLoadGlobalAddress);
  loadGlobalAddrInst->set_operand(0, base);
  loadGlobalAddrInst->set_operand(1, MIROperand::as_reloc(mFloatDataStorage));
  ctx.emit_inst(loadGlobalAddrInst);
  
  // Add addr, base, offset
  const auto addr = ctx.new_vreg(ptrType);
  auto addInst = new MIRInst(InstAdd);
  addInst->set_operand(0, addr);
  addInst->set_operand(1, base);
  addInst->set_operand(2, MIROperand::as_imm(offset, ptrType));
  ctx.emit_inst(addInst);

  // Load dst, addr, 4
  const auto dst = ctx.new_vreg(OperandType::Float32);
  auto loadInst = new MIRInst(InstLoad);
  loadInst->set_operand(0, dst);
  loadInst->set_operand(1, addr);
  loadInst->set_operand(2, MIROperand::as_imm(4, OperandType::Special));    
  ctx.emit_inst(loadInst);  
  return dst;
}


static OperandType get_optype(ir::Type* type) {
  if (type->isInt()) {
    switch (type->btype()) {
      case ir::INT1: return OperandType::Bool;
      case ir::INT32: return OperandType::Int32;
      default: assert(false && "unsupported int type");
    }
  } else if (type->isFloatPoint()) {
    switch (type->btype()) {
      case ir::FLOAT: return OperandType::Float32;
      default: assert(false && "unsupported float type");
    }
  } else if (type->isPointer()) {  // NOTE: rv64
    return OperandType::Int64;
  } else {
    return OperandType::Special;
  }
}

MIROperand* LoweringContext::new_vreg(ir::Type* type) {
  auto optype = get_optype(type);
  return MIROperand::as_vreg(_code_gen_ctx->next_id(), optype);
}

MIROperand* LoweringContext::new_vreg(OperandType type) {
  return MIROperand::as_vreg(_code_gen_ctx->next_id(), type);
}

void LoweringContext::add_valmap(ir::Value* ir_val, MIROperand* mir_operand) {
  if (_val_map.count(ir_val)) assert(false && "value already mapped");
  _val_map.emplace(ir_val, mir_operand);
}

MIROperand* LoweringContext::map2operand(ir::Value* ir_val) {
  /* 1. Local Value */
  auto iter = _val_map.find(ir_val);
  if (iter != _val_map.end()) return new MIROperand(*(iter->second));

  /* 2. Global Value */
  if (auto gvar = dyn_cast<ir::GlobalVariable>(ir_val)) {
    auto ptr = new_vreg(_ptr_type);
    auto inst = new MIRInst(InstLoadGlobalAddress);
    auto g_reloc = gvar_map.at(gvar)->reloc.get();  // MIRRelocable*
    auto operand = MIROperand::as_reloc(g_reloc);
    inst->set_operand(0, ptr);
    inst->set_operand(1, operand);
    emit_inst(inst);
    return ptr;
  }

  /* 3. Constant */
  if (not ir::isa<ir::Constant>(ir_val)) {
    std::cerr << "error: " << ir_val->name() << " must be constant"
              << std::endl;
    assert(false);
  }

  auto const_val = dyn_cast<ir::Constant>(ir_val);
  if (const_val->type()->isInt()) {
    auto imm = MIROperand::as_imm(const_val->i32(), OperandType::Int32);
    return imm;
  }
  // TODO: support float constant
  if (const_val->type()->isFloat32()) {
    if (auto fpOperand = _code_gen_ctx->iselInfo->materializeFPConstant(
          const_val->f32(), *this)) {
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
  return nullptr;
}

void LoweringContext::emit_inst(MIRInst* inst) {
  _mir_block->insts().emplace_back(inst);
}
void LoweringContext::emit_copy(MIROperand* dst, MIROperand* src) {
  auto inst = new MIRInst(select_copy_opcode(dst, src));
  inst->set_operand(0, dst);
  inst->set_operand(1, src);
  emit_inst(inst);
}

void create_mir_module(ir::Module& ir_module, MIRModule& mir_module, Target& target, 
                       pass::topAnalysisInfoManager* tAIM);
MIRFunction* create_mir_function(ir::Function* ir_func, MIRFunction* mir_func,
                                 CodeGenContext& codegen_ctx, LoweringContext& lowering_ctx, 
                                 pass::topAnalysisInfoManager* tAIM);
MIRInst* create_mir_inst(ir::Instruction* ir_inst, LoweringContext& ctx);
void lower_GetElementPtr(ir::inst_iterator begin, ir::inst_iterator end, LoweringContext& ctx);

std::unique_ptr<MIRModule> create_mir_module(ir::Module& ir_module, Target& target,
                                             pass::topAnalysisInfoManager* tAIM) {
    auto mir_module_uptr = std::make_unique<MIRModule>(&ir_module, target);
    create_mir_module(ir_module, *mir_module_uptr, target, tAIM);
    return mir_module_uptr;
}

union FloatUint32 {
    float f;
    uint32_t u;
} fu32;

void create_mir_module(ir::Module& ir_module, MIRModule& mir_module, Target& target, pass::topAnalysisInfoManager* tAIM) {
    constexpr bool debugLowering = true;

    auto& functions = mir_module.functions();
    auto& global_objs = mir_module.global_objs();
    LoweringContext lowering_ctx(mir_module, target);
    auto& func_map = lowering_ctx.func_map;
    auto& gvar_map = lowering_ctx.gvar_map;

    //! 1. for all functions, create MIRFunction
    for (auto func : ir_module.funcs()) {
        functions.push_back(std::make_unique<MIRFunction>(func->name(), &mir_module));
        func_map.emplace(func, functions.back().get());
    }

    //! 2. for all global variables, create MIRGlobalObject
    for (auto ir_gvar : ir_module.globalVars()) {
        auto name = ir_gvar->name().substr(1);  /* remove '@' */
        /* 基础类型 (int OR float) */
        auto type = dyn_cast<ir::PointerType>(ir_gvar->type())->baseType();
        if (type->isArray()) type = dyn_cast<ir::ArrayType>(type)->baseType();
        const size_t size = type->size();
        const bool read_only = ir_gvar->isConst();
        const bool is_float = type->isFloat32();
        const size_t align = 4;

        if (ir_gvar->isInit()) {  /* .data: 已初始化的、可修改的全局数据 (Array and Scalar) */
            /* 全局变量初始化一定为常值表达式 */
            MIRDataStorage::Storage data; const auto idx = ir_gvar->init_cnt();
            for (int i = 0; i < idx; i++) {
                auto val = dyn_cast<ir::Constant>(ir_gvar->init(i));
                /* NOTE: float to uint32_t, type cast, doesn't change the memory */
                if (type->isInt()) {
                    fu32.u = val->i32();
                } else if (type->isFloat32()) {
                    fu32.f = val->f32();
                } else {
                    assert(false && "Not Supported Type.");
                }
                data.push_back(fu32.u);
            }
            auto mir_storage = std::make_unique<MIRDataStorage>(std::move(data), read_only, name, is_float);
            auto mir_gobj = std::make_unique<MIRGlobalObject>(align, std::move(mir_storage), &mir_module);
            mir_module.global_objs().push_back(std::move(mir_gobj));
        } else {  /* .bss: 未初始化的全局数据 (Just Scalar) */
            auto mir_storage = std::make_unique<MIRZeroStorage>(size, name, is_float);
            auto mir_gobj = std::make_unique<MIRGlobalObject>(align, std::move(mir_storage), &mir_module);
            mir_module.global_objs().push_back(std::move(mir_gobj));
        }
        gvar_map.emplace(ir_gvar, mir_module.global_objs().back().get());
    }

    // TODO: transformModuleBeforeCodeGen

    //! 3. codegen
    CodeGenContext codegen_ctx{target, target.get_datalayout(),
                               target.get_target_inst_info(),
                               target.get_target_frame_info(), MIRFlags{}};
    codegen_ctx.iselInfo = &target.get_target_isel_info();
    codegen_ctx.scheduleModel = &target.get_schedule_model();
    lowering_ctx._code_gen_ctx = &codegen_ctx;
    IPRAUsageCache infoIPRA;

    auto dumpStageWithMsg = [&](std::ostream& os, std::string_view stage, std::string_view msg) {
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


    //! 4. lower all functions
    for (auto& ir_func : ir_module.funcs()) {
        size_t stageIdx = 0;

        auto dumpStageResult = [&stageIdx](std::string stage, MIRFunction* mir_func, CodeGenContext& codegen_ctx) {
            if (not debugLowering) return;
            auto fileName = mir_func->name() +  std::to_string(stageIdx) + "_" + stage + ".ll";
            std::ofstream fout("./.debug/" + fileName);
            mir_func->print(fout, codegen_ctx);
            stageIdx++;
        };
        auto mir_func = func_map[ir_func];
        if (ir_func->blocks().empty()) continue;
        if (debugLowering) {
            auto fileName = mir_func->name()  +  std::to_string(stageIdx) + "_" +  "BeforeLowering.ll";
            std::ofstream fout("./.debug/" + fileName);
            ir_func->print(fout);
            stageIdx++;
        }

        /* 4.1: lower function body to generic MIR */
        {
            create_mir_function(ir_func, mir_func, codegen_ctx, lowering_ctx, tAIM);
            if(not mir_func->verify(std::cerr, codegen_ctx)) { 
                std::cerr << "Lowering Error: " << mir_func->name() << " failed to verify." << std::endl;
            }
            dumpStageResult("AfterLowering", mir_func, codegen_ctx);
        }

        /* 4.2: instruction selection */
        {
            ISelContext isel_ctx(codegen_ctx);
            isel_ctx.run_isel(mir_func);    
            dumpStageResult("AfterIsel", mir_func, codegen_ctx);
        }
        /* 4.3 Optimize: register coalescing */

        /* 4.4 Optimize: peephole optimization (窥孔优化) */
        // while (genericPeepholeOpt(*mir_func, codegen_ctx));

        /* 4.5 pre-RA legalization */

        /* 4.6 Optimize: pre-RA scheduling, minimize register usage */
        // {
        //     preRASchedule(*mir_func, codegen_ctx);
        //     dumpStageResult("AfterPreRASchedule", mir_func, codegen_ctx);
        // }

        /* 4.7 register allocation */
        {
            codegen_ctx.registerInfo = new RISCVRegisterInfo();
            if (codegen_ctx.registerInfo) {
                GraphColoringAllocate(*mir_func, codegen_ctx, infoIPRA);
                // fastAllocator(*mir_func, codegen_ctx, infoIPRA);
                dumpStageResult("AfterGraphColoring", mir_func, codegen_ctx);
            }
        }

        /* 4.8 stack allocation */
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
        /* 4.10 post legalization */
        postLegalizeFunc(*mir_func, codegen_ctx);
        
        /* 4.11 verify */
        
        /* 4.12 Add Function to IPRA cache */
        if(codegen_ctx.registerInfo) {
            infoIPRA.add(codegen_ctx, *mir_func);
        }

        dumpStageResult("AfterCodeGen", mir_func, codegen_ctx);
    }
    /* module verify */
}

MIRFunction* create_mir_function(ir::Function* ir_func, MIRFunction* mir_func,
                                 CodeGenContext& codegen_ctx, LoweringContext& lowering_ctx, 
                                 pass::topAnalysisInfoManager* tAIM) {
    
    lowering_ctx.set_mir_func(mir_func);
    /* Some Debug Information */
    constexpr bool debugLowerFunc = false;
    constexpr bool DebugCreateMirFunction = false;
    // TODO: before lowering, get some analysis pass result

    //! 1. map from ir to mir
    auto& block_map = lowering_ctx._block_map;
    std::unordered_map<ir::Value*, MIROperand*> storage_map;
    auto& target = codegen_ctx.target;
    auto& datalayout = target.get_datalayout();

    for (auto ir_block : ir_func->blocks()) {
        mir_func->blocks().push_back(std::make_unique<MIRBlock>(mir_func, "label" + std::to_string(codegen_ctx.next_id_label())));
        block_map.emplace(ir_block, mir_func->blocks().back().get());
    }
    if (DebugCreateMirFunction) std::cerr << "stage 1: map from it to mir" << std::endl;

    //! 2. emitPrologue for function
    {
        /* assign vreg to arg */
        for (auto ir_arg : ir_func->args()) {
            auto vreg = lowering_ctx.new_vreg(ir_arg->type());
            lowering_ctx.add_valmap(ir_arg, vreg);
            mir_func->args().push_back(vreg);
        }
        lowering_ctx.set_mir_block(block_map.at(ir_func->entry()));
        codegen_ctx.frameInfo.emit_prologue(mir_func, lowering_ctx);
    }
    if (DebugCreateMirFunction) std::cerr << "stage 2: emitPrologue for function" << std::endl;

    //! 3. process alloca, new stack object for each alloca
    lowering_ctx.set_mir_block(block_map.at(ir_func->entry()));  // entry
    for (auto& ir_inst : ir_func->entry()->insts()) {  // NOTE: all alloca in entry
        if (ir_inst->valueId() != ir::ValueId::vALLOCA) continue;

        auto pointee_type = dyn_cast<ir::PointerType>(ir_inst->type())->baseType();
        uint32_t align = 4;  // TODO: align, need bind to ir object
        auto storage = mir_func->add_stack_obj(
            codegen_ctx.next_id(),                        // id
            static_cast<uint32_t>(pointee_type->size()),  // size
            align,                                        // align
            0,                                            // offset
            StackObjectUsage::Local);
        storage_map.emplace(ir_inst, storage);
        // emit load stack object addr inst
        auto addr = lowering_ctx.new_vreg(lowering_ctx.get_ptr_type());
        auto ldsa_inst = new MIRInst{InstLoadStackObjectAddr};
        ldsa_inst->set_operand(0, addr);
        ldsa_inst->set_operand(1, storage);
        lowering_ctx.emit_inst(ldsa_inst);
        // map
        lowering_ctx.add_valmap(ir_inst, addr);
    }
    if (DebugCreateMirFunction) std::cerr << "stage 3: process alloca instruction, new stack object for each alloca instruction" << std::endl;

    //! 4. lowering all blocks
    {
        for (auto ir_block : ir_func->blocks()) {
            auto mir_block = block_map[ir_block];
            lowering_ctx.set_mir_block(mir_block);
            auto& instructions = ir_block->insts();
            for (auto iter = instructions.begin(); iter != instructions.end();) {
                auto ir_inst = *iter;
                if (ir_inst->valueId() == ir::ValueId::vALLOCA) iter++;
                else if (ir_inst->valueId() == ir::ValueId::vGETELEMENTPTR) {
                    auto end = iter; end++;
                    while (end != instructions.end() && (*end)->valueId() == ir::ValueId::vGETELEMENTPTR) {
                        auto preInst = std::prev(end);
                        if (dyn_cast<ir::GetElementPtrInst>(*end)->value() == (*preInst)) end++;
                        else break;
                    }
                    lower_GetElementPtr(iter, end, lowering_ctx);
                    iter = end;
                } else {
                    create_mir_inst(ir_inst, lowering_ctx);
                    iter++;
                }
                if (debugLowerFunc) {
                    ir_inst->print(std::cerr);
                    std::cerr << std::endl;
                }
            }
        }
    }
    if (DebugCreateMirFunction) std::cerr << "stage 4: lowering all blocks" << std::endl;
    return mir_func;
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

MIRInst* create_mir_inst(ir::Instruction* ir_inst, LoweringContext& ctx) {
    switch (ir_inst->valueId()) {
        case ir::ValueId::vFNEG:
        case ir::ValueId::vTRUNC:
        case ir::ValueId::vZEXT:
        case ir::ValueId::vSEXT:
        case ir::ValueId::vFPTRUNC:
        case ir::ValueId::vFPTOSI:
        case ir::ValueId::vSITOFP: lower(dyn_cast<ir::UnaryInst>(ir_inst), ctx); break;
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
        case ir::ValueId::vFREM: lower(dyn_cast<ir::BinaryInst>(ir_inst), ctx); break;
        case ir::ValueId::vIEQ:
        case ir::ValueId::vINE:
        case ir::ValueId::vISGT:
        case ir::ValueId::vISGE:
        case ir::ValueId::vISLT:
        case ir::ValueId::vISLE: lower(dyn_cast<ir::ICmpInst>(ir_inst), ctx); break;
        case ir::ValueId::vFOEQ:
        case ir::ValueId::vFONE:
        case ir::ValueId::vFOGT:
        case ir::ValueId::vFOGE:
        case ir::ValueId::vFOLT:
        case ir::ValueId::vFOLE: lower(dyn_cast<ir::FCmpInst>(ir_inst), ctx); break;
        case ir::ValueId::vALLOCA: std::cerr << "alloca not supported" << std::endl; break;
        case ir::ValueId::vLOAD: lower(dyn_cast<ir::LoadInst>(ir_inst), ctx); break;
        case ir::ValueId::vSTORE: lower(dyn_cast<ir::StoreInst>(ir_inst), ctx); break;
        case ir::ValueId::vGETELEMENTPTR: assert(false && "not supported inst");
        case ir::ValueId::vRETURN: lower(dyn_cast<ir::ReturnInst>(ir_inst), ctx); break;
        case ir::ValueId::vBR: lower(dyn_cast<ir::BranchInst>(ir_inst), ctx); break;
        case ir::ValueId::vCALL: lower(dyn_cast<ir::CallInst>(ir_inst), ctx); break;
        case ir::ValueId::vMEMSET: lower(dyn_cast<ir::MemsetInst>(ir_inst), ctx); break;
        case ir::ValueId::vBITCAST: lower(dyn_cast<ir::BitCastInst>(ir_inst), ctx); break;
        default:
            const auto valueIdEnumName = utils::enumName(static_cast<ir::ValueId>(ir_inst->valueId()));
            std::cerr << valueIdEnumName << ": not supported inst" << std::endl;
            assert(false && "not supported inst");
    }
    return nullptr;
}
void lower(ir::UnaryInst* ir_inst, LoweringContext& ctx) {
    auto gc_instid = [scid = ir_inst->valueId()] {
        switch (scid) {
            case ir::ValueId::vFNEG: return InstFNeg;
            case ir::ValueId::vTRUNC: return InstTrunc;
            case ir::ValueId::vZEXT: return InstZExt;
            case ir::ValueId::vSEXT: return InstSExt;
            case ir::ValueId::vFPTRUNC: assert(false && "not supported unary inst");
            case ir::ValueId::vFPTOSI: return InstF2S;
            case ir::ValueId::vSITOFP: return InstS2F;
            default: assert(false && "not supported unary inst");
        }
    }();

    auto ret = ctx.new_vreg(ir_inst->type());
    auto inst = new MIRInst(gc_instid);
    inst->set_operand(0, ret); 
    inst->set_operand(1, ctx.map2operand(ir_inst->operand(0)));
    ctx.emit_inst(inst);
    ctx.add_valmap(ir_inst, ret);
}

/*
 * @brief: Lowering ICmpInst (int OR float)
 * @note: 
 *      1. int
 *          IR: <result> = icmp <cond> <ty> <op1>, <op2>
 *          MIRGeneric: ICmp dst, src1, src2, op
 *      2. float
 *          IR: <result> = fcmp [fast-math flags]* <cond> <ty> <op1>, <op2>
 *          MIRGeneric: FCmp dst, src1, src2, op
 */
void lower(ir::ICmpInst* ir_inst, LoweringContext& ctx) {
    // TODO: Need Test
    auto op = [scid = ir_inst->valueId()] {
        switch (scid) {
            case ir::ValueId::vIEQ: return CompareOp::ICmpEqual;
            case ir::ValueId::vINE: return CompareOp::ICmpNotEqual;
            case ir::ValueId::vISGT: return CompareOp::ICmpSignedGreaterThan;
            case ir::ValueId::vISGE: return CompareOp::ICmpSignedGreaterEqual;
            case ir::ValueId::vISLT: return CompareOp::ICmpSignedLessThan;
            case ir::ValueId::vISLE: return CompareOp::ICmpSignedLessEqual;
            default: assert(false && "not supported icmp inst");
        }
    }();

    auto ret = ctx.new_vreg(ir_inst->type());
    auto inst = new MIRInst(InstICmp);
    inst->set_operand(0, ret);
    inst->set_operand(1, ctx.map2operand(ir_inst->operand(0)));
    inst->set_operand(2, ctx.map2operand(ir_inst->operand(1)));
    inst->set_operand(3, MIROperand::as_imm(static_cast<uint32_t>(op), OperandType::Special));
    ctx.emit_inst(inst);
    ctx.add_valmap(ir_inst, ret);
}
void lower(ir::FCmpInst* ir_inst, LoweringContext& ctx) {
    auto op = [scid = ir_inst->valueId()] {
        switch (scid) {
            case ir::ValueId::vFOEQ: return CompareOp::FCmpOrderedEqual;
            case ir::ValueId::vFONE: return CompareOp::FCmpOrderedNotEqual;
            case ir::ValueId::vFOGT: return CompareOp::FCmpOrderedGreaterThan;
            case ir::ValueId::vFOGE: return CompareOp::FCmpOrderedGreaterEqual;
            case ir::ValueId::vFOLT: return CompareOp::FCmpOrderedLessThan;
            case ir::ValueId::vFOLE: return CompareOp::FCmpOrderedLessEqual;
            default: assert(false && "not supported fcmp inst");
        }
    }();

    auto ret = ctx.new_vreg(ir_inst->type());
    auto inst = new MIRInst(InstFCmp);
    inst->set_operand(0, ret);
    inst->set_operand(1, ctx.map2operand(ir_inst->operand(0)));
    inst->set_operand(2, ctx.map2operand(ir_inst->operand(1)));
    inst->set_operand(3, MIROperand::as_imm(static_cast<uint32_t>(op), OperandType::Special));
    ctx.emit_inst(inst);
    ctx.add_valmap(ir_inst, ret);
}

/* CallInst */
void lower(ir::CallInst* ir_inst, LoweringContext& ctx) {
    ctx._target.get_target_frame_info().emit_call(ir_inst, ctx);
}
/* BinaryInst */
void lower(ir::BinaryInst* ir_inst, LoweringContext& ctx) {
    auto gc_instid = [scid = ir_inst->valueId()] {
        switch (scid) {
            case ir::ValueId::vADD: return InstAdd;
            case ir::ValueId::vFADD: return InstFAdd;
            case ir::ValueId::vSUB: return InstSub;
            case ir::ValueId::vFSUB: return InstFSub;
            case ir::ValueId::vMUL: return InstMul;
            case ir::ValueId::vFMUL: return InstFMul;
            case ir::ValueId::vUDIV: return InstUDiv;
            case ir::ValueId::vSDIV: return InstSDiv;
            case ir::ValueId::vFDIV: return InstFDiv;
            case ir::ValueId::vUREM: return InstURem;
            case ir::ValueId::vSREM: return InstSRem;
            default: assert(false && "not supported binary inst");
        }
    }();

    auto ret = ctx.new_vreg(ir_inst->type());
    auto inst = new MIRInst(gc_instid);
    inst->set_operand(0, ret);
    inst->set_operand(1, ctx.map2operand(ir_inst->operand(0)));
    inst->set_operand(2, ctx.map2operand(ir_inst->operand(1)));
    ctx.emit_inst(inst);
    ctx.add_valmap(ir_inst, ret);
}

/* BranchInst */
void emit_branch(ir::BasicBlock* srcblock, ir::BasicBlock* dstblock, LoweringContext& lctx);
void lower(ir::BranchInst* ir_inst, LoweringContext& ctx) {
    auto src_block = ir_inst->block();
    auto mir_func = ctx._mir_func;
    const auto codegen_ctx = ctx._code_gen_ctx;
    if (ir_inst->is_cond()) {
        // conditional branch
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
        auto inst = new MIRInst(InstBranch);
        inst->set_operand(0, ctx.map2operand(ir_inst->cond()));
        inst->set_operand(1, MIROperand::as_reloc(ctx.map2block(ir_inst->iftrue())));
        inst->set_operand(2, MIROperand::as_prob(0.5));
        ctx.emit_inst(inst);
        auto findBlockIter = [mir_func] (const MIRBlock* block) {
            return std::find_if(mir_func->blocks().begin(), mir_func->blocks().end(), 
                [block](const std::unique_ptr<MIRBlock>& mir_block) { return mir_block.get() == block; });
        };
        {
        auto curBlockIter = findBlockIter(ctx.get_mir_block());
        assert(curBlockIter != mir_func->blocks().end());

        auto newBlock = std::make_unique<MIRBlock>(ctx._mir_func, "label" + std::to_string(codegen_ctx->next_id_label()));
        auto newBlockPtr = newBlock.get();
        // insert new block after current block
        mir_func->blocks().insert(++curBlockIter, std::move(newBlock));
        ctx.set_mir_block(newBlockPtr);
        }
        auto jump_inst = new MIRInst(InstJump);
        jump_inst->set_operand(0, MIROperand::as_reloc(ctx.map2block(ir_inst->iffalse())));
        ctx.emit_inst(jump_inst);
    } else {  // unconditional branch
        auto dst_block = ir_inst->dest();
        emit_branch(src_block, dst_block, ctx);
    }
}
void emit_branch(ir::BasicBlock* srcblock, ir::BasicBlock* dstblock, LoweringContext& lctx) {
    auto dst_mblock = lctx.map2block(dstblock);
    auto src_mblock = lctx.map2block(srcblock);
    auto operand = MIROperand::as_reloc(dst_mblock);
    auto inst = new MIRInst(InstJump);
    inst->set_operand(0, operand);
    lctx.emit_inst(inst);
}

/* LoadInst */
void lower(ir::LoadInst* ir_inst, LoweringContext& ctx) {
    auto ret = ctx.new_vreg(ir_inst->type()); auto ptr = ctx.map2operand(ir_inst->operand(0));
    assert(ret != nullptr && ptr != nullptr);
    const uint32_t align = 4;
    auto inst = new MIRInst(InstLoad);
    inst->set_operand(0, ret); inst->set_operand(1, ptr);
    inst->set_operand(2, MIROperand::as_imm(align, OperandType::Alignment));
    ctx.emit_inst(inst);
    ctx.add_valmap(ir_inst, ret);
}
/* StoreInst */
void lower(ir::StoreInst* ir_inst, LoweringContext& ctx) {
    auto inst = new MIRInst(InstStore);
    inst->set_operand(0, ctx.map2operand(ir_inst->ptr()));
    inst->set_operand(1, ctx.map2operand(ir_inst->value()));
    const uint32_t align = 4;
    inst->set_operand(2, MIROperand::as_imm(align, OperandType::Alignment));
    ctx.emit_inst(inst);
}
/* ReturnInst */
void lower(ir::ReturnInst* ir_inst, LoweringContext& ctx) {
    ctx._target.get_target_frame_info().emit_return(ir_inst, ctx);
}
/* BitCastInst */
void lower(ir::BitCastInst* ir_inst, LoweringContext& ctx) {
    const auto ir_bitcast_inst = dyn_cast<ir::BitCastInst>(ir_inst);
    const auto base = ir_bitcast_inst->value();
    ctx.add_valmap(ir_bitcast_inst, ctx.map2operand(base));
}
/* MemsetInst */
void lower(ir::MemsetInst* ir_inst, LoweringContext& ctx) {
    const auto ir_memset_inst = dyn_cast<ir::MemsetInst>(ir_inst);
    const auto ir_pointer = ir_memset_inst->value();
    const auto size = dyn_cast<ir::PointerType>(ir_pointer->type())->baseType()->size();

    /* 通过寄存器传递参数 */
    // 1. 指针
    {
        auto val = ctx.map2operand(ir_pointer);
        MIROperand* dst = MIROperand::as_isareg(RISCV::X10, OperandType::Int64);
        assert(dst);
        ctx.emit_copy(dst, val);
    }
    
    // 2. 长度
    {
        auto val = ctx.map2operand(ir::Constant::gen_i32<int>(size));
        MIROperand* dst = MIROperand::as_isareg(RISCV::X11, OperandType::Int64);
        assert(dst);
        ctx.emit_copy(dst, val);
    }

    /* 生成跳转至被调用函数的指令 */
    {
        auto callInst = new MIRInst(RISCV::JAL);
        callInst->set_operand(0, MIROperand::as_reloc(ctx.memsetFunc));
        ctx.emit_inst(callInst);
    }
}
/*
 * @brief: lower GetElementPtrInst [begin, end)
 * @note: 
 *      1. Array: <result> = getelementptr <type>, <type>* <ptrval>, i32 0, i32 <idx>
 *      2. Pointer: <result> = getelementptr <type>, <type>* <ptrval>, i32 <idx>
 */
void lower_GetElementPtr(ir::inst_iterator begin, ir::inst_iterator end, LoweringContext& ctx) {
    const auto dims = dyn_cast<ir::GetElementPtrInst>(*begin)->cur_dims();
    // const auto dims_cnt = dyn_cast<ir::GetElementPtrInst>(*begin)->cur_dims_cnt();
    size_t dims_cnt = 0;
    const int begin_id = dyn_cast<ir::GetElementPtrInst>(*begin)->getid();
    const auto base = ctx.map2operand(dyn_cast<ir::GetElementPtrInst>(*begin)->value());  // 基地址

    /* 统计GetElementPtr指令数量 */
    auto iter = begin; ir::Value* instEnd = nullptr; MIROperand* ptr = base;
    std::vector<ir::Value*> idx;
    while (iter != end) {
        idx.push_back(dyn_cast<ir::GetElementPtrInst>(*iter)->index());
        instEnd = *iter;
        iter++; dims_cnt++;
    }
    dims_cnt = std::min(dims_cnt, dyn_cast<ir::GetElementPtrInst>(*begin)->cur_dims_cnt());

    /* 计算偏移量 */
    int dimension = 0;
    MIROperand* mir_offset = nullptr; auto ir_offset = idx[dimension++];
    bool is_constant = dyn_cast<ir::Constant>(ir_offset) ? true : false;
    if (!is_constant) mir_offset = ctx.map2operand(ir_offset);
    if (begin_id == 0) {  // 指针
        if (mir_offset) {
            auto newPtr = ctx.new_vreg(OperandType::Int32);
            auto newInst = new MIRInst(InstMul);
            newInst->set_operand(0, newPtr); newInst->set_operand(1, mir_offset); newInst->set_operand(2, MIROperand::as_imm<int>(4, OperandType::Int32));
            ctx.emit_inst(newInst);
            mir_offset = newPtr;
        } else {
            auto ir_offset_constant = dyn_cast<ir::Constant>(ir_offset);
            mir_offset = ctx.map2operand(ir::Constant::gen_i32<int>(ir_offset_constant->i32() * 4));
        }
        auto newPtr = ctx.new_vreg(OperandType::Int64);
        auto newInst = new MIRInst(InstAdd);
        newInst->set_operand(0, newPtr); newInst->set_operand(1, ptr); newInst->set_operand(2, mir_offset);
        ctx.emit_inst(newInst);
        ptr = newPtr;
        ctx.add_valmap(instEnd, ptr);
    }

    if (dimension <= dims_cnt) {
        for (; dimension < dims_cnt; dimension++) {
            /* 乘法 */
            const auto alpha = dims[dimension];  // int
            if (is_constant) {  // 常量
                const auto ir_offset_constant = dyn_cast<ir::Constant>(ir_offset);
                ir_offset = ir::Constant::gen_i32<int>(ir_offset_constant->i32() * alpha);
            } else {  // 变量
                auto newPtr = ctx.new_vreg(OperandType::Int32);
                auto newInst = new MIRInst(InstMul);
                newInst->set_operand(0, newPtr); newInst->set_operand(1, mir_offset); newInst->set_operand(2, MIROperand::as_imm<int>(alpha, OperandType::Int32));
                ctx.emit_inst(newInst);
                mir_offset = newPtr;
            }

            /* 加法 */
            auto ir_current_idx = idx[dimension];  // ir::Value*
            if (is_constant && dyn_cast<ir::Constant>(ir_current_idx)) {
                const auto ir_current_idx_constant = dyn_cast<ir::Constant>(ir_current_idx);
                const auto ir_offset_constant = dyn_cast<ir::Constant>(ir_offset);
                ir_offset = ir::Constant::gen_i32<int>(ir_current_idx_constant->i32() + ir_offset_constant->i32());
            } else {
                if (is_constant) mir_offset = ctx.map2operand(ir_offset);
                is_constant = false;
                auto newPtr = ctx.new_vreg(OperandType::Int32);
                auto newInst = new MIRInst(InstAdd);
                newInst->set_operand(0, newPtr); newInst->set_operand(1, mir_offset);
                newInst->set_operand(2, ctx.map2operand(ir_current_idx));
                ctx.emit_inst(newInst);
                mir_offset = newPtr;
            }
        }

        if (mir_offset) {
            auto newPtr = ctx.new_vreg(OperandType::Int32);
            auto newInst = new MIRInst(InstMul);
            newInst->set_operand(0, newPtr); newInst->set_operand(1, mir_offset);
            newInst->set_operand(2, MIROperand::as_imm<int>(4, OperandType::Int32));
            ctx.emit_inst(newInst);
            mir_offset = newPtr;
        } else {
            auto ir_offset_constant = dyn_cast<ir::Constant>(ir_offset);
            mir_offset = ctx.map2operand(ir::Constant::gen_i32<int>(ir_offset_constant->i32() * 4));
        }
        auto newPtr = ctx.new_vreg(OperandType::Int64);
        auto newInst = new MIRInst(InstAdd);
        newInst->set_operand(0, newPtr); newInst->set_operand(1, ptr); newInst->set_operand(2, mir_offset);
        ctx.emit_inst(newInst);
        ptr = newPtr;
        ctx.add_valmap(instEnd, ptr);
    }
}
}