; ModuleID = 'parallelFor.cpp'
source_filename = "parallelFor.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

%"class.std::ios_base::Init" = type { i8 }
%"class.std::basic_ostream" = type { i32 (...)**, %"class.std::basic_ios" }
%"class.std::basic_ios" = type { %"class.std::ios_base", %"class.std::basic_ostream"*, i8, i8, %"class.std::basic_streambuf"*, %"class.std::ctype"*, %"class.std::num_put"*, %"class.std::num_get"* }
%"class.std::ios_base" = type { i32 (...)**, i64, i64, i32, i32, i32, %"struct.std::ios_base::_Callback_list"*, %"struct.std::ios_base::_Words", [8 x %"struct.std::ios_base::_Words"], i32, %"struct.std::ios_base::_Words"*, %"class.std::locale" }
%"struct.std::ios_base::_Callback_list" = type { %"struct.std::ios_base::_Callback_list"*, void (i32, %"class.std::ios_base"*, i32)*, i32, i32 }
%"struct.std::ios_base::_Words" = type { i8*, i64 }
%"class.std::locale" = type { %"class.std::locale::_Impl"* }
%"class.std::locale::_Impl" = type { i32, %"class.std::locale::facet"**, i64, %"class.std::locale::facet"**, i8** }
%"class.std::locale::facet" = type <{ i32 (...)**, i32, [4 x i8] }>
%"class.std::basic_streambuf" = type { i32 (...)**, i8*, i8*, i8*, i8*, i8*, i8*, %"class.std::locale" }
%"class.std::ctype" = type <{ %"class.std::locale::facet.base", [4 x i8], %struct.__locale_struct*, i8, [7 x i8], i32*, i32*, i16*, i8, [256 x i8], [256 x i8], i8, [6 x i8] }>
%"class.std::locale::facet.base" = type <{ i32 (...)**, i32 }>
%struct.__locale_struct = type { [13 x %struct.__locale_data*], i16*, i32*, i32*, [13 x i8*] }
%struct.__locale_data = type opaque
%"class.std::num_put" = type { %"class.std::locale::facet.base", [4 x i8] }
%"class.std::num_get" = type { %"class.std::locale::facet.base", [4 x i8] }
%"class.std::unique_ptr" = type { %"struct.std::__uniq_ptr_data" }
%"struct.std::__uniq_ptr_data" = type { %"class.std::__uniq_ptr_impl" }
%"class.std::__uniq_ptr_impl" = type { %"class.std::tuple" }
%"class.std::tuple" = type { %"struct.std::_Tuple_impl" }
%"struct.std::_Tuple_impl" = type { %"struct.std::_Head_base.1" }
%"struct.std::_Head_base.1" = type { %"struct.std::thread::_State"* }
%"struct.std::thread::_State" = type { i32 (...)** }
%"class.std::vector" = type { %"struct.std::_Vector_base" }
%"struct.std::_Vector_base" = type { %"struct.std::_Vector_base<std::thread, std::allocator<std::thread>>::_Vector_impl" }
%"struct.std::_Vector_base<std::thread, std::allocator<std::thread>>::_Vector_impl" = type { %"struct.std::_Vector_base<std::thread, std::allocator<std::thread>>::_Vector_impl_data" }
%"struct.std::_Vector_base<std::thread, std::allocator<std::thread>>::_Vector_impl_data" = type { %"class.std::thread"*, %"class.std::thread"*, %"class.std::thread"* }
%"class.std::thread" = type { %"class.std::thread::id" }
%"class.std::thread::id" = type { i64 }
%"struct.std::thread::_State_impl" = type { %"struct.std::thread::_State", %"struct.std::thread::_Invoker" }
%"struct.std::thread::_Invoker" = type { %"class.std::tuple.2" }
%"class.std::tuple.2" = type { %"struct.std::_Tuple_impl.3" }
%"struct.std::_Tuple_impl.3" = type { %"struct.std::_Tuple_impl.4", %"struct.std::_Head_base.8" }
%"struct.std::_Tuple_impl.4" = type { %"struct.std::_Tuple_impl.5", %"struct.std::_Head_base.7" }
%"struct.std::_Tuple_impl.5" = type { %"struct.std::_Head_base.6" }
%"struct.std::_Head_base.6" = type { i32 }
%"struct.std::_Head_base.7" = type { i32 }
%"struct.std::_Head_base.8" = type { void (i32, i32)* }

$_ZNSt6vectorISt6threadSaIS0_EED2Ev = comdat any

$__clang_call_terminate = comdat any

$_ZNSt6vectorISt6threadSaIS0_EE17_M_realloc_insertIJRPFviiERiS7_EEEvN9__gnu_cxx17__normal_iteratorIPS0_S2_EEDpOT_ = comdat any

$_ZNSt6thread11_State_implINS_8_InvokerISt5tupleIJPFviiEiiEEEEED0Ev = comdat any

$_ZNSt6thread11_State_implINS_8_InvokerISt5tupleIJPFviiEiiEEEEE6_M_runEv = comdat any

$_ZTVNSt6thread11_State_implINS_8_InvokerISt5tupleIJPFviiEiiEEEEEE = comdat any

$_ZTSNSt6thread11_State_implINS_8_InvokerISt5tupleIJPFviiEiiEEEEEE = comdat any

$_ZTINSt6thread11_State_implINS_8_InvokerISt5tupleIJPFviiEiiEEEEEE = comdat any

@_ZStL8__ioinit = internal global %"class.std::ios_base::Init" zeroinitializer, align 1
@__dso_handle = external hidden global i8
@_ZSt4cerr = external global %"class.std::basic_ostream", align 8
@.str = private unnamed_addr constant [18 x i8] c"Launching thread \00", align 1
@.str.1 = private unnamed_addr constant [14 x i8] c" with range [\00", align 1
@.str.2 = private unnamed_addr constant [3 x i8] c", \00", align 1
@.str.3 = private unnamed_addr constant [3 x i8] c")\0A\00", align 1
@_ZTVNSt6thread11_State_implINS_8_InvokerISt5tupleIJPFviiEiiEEEEEE = linkonce_odr dso_local unnamed_addr constant { [5 x i8*] } { [5 x i8*] [i8* null, i8* bitcast ({ i8*, i8*, i8* }* @_ZTINSt6thread11_State_implINS_8_InvokerISt5tupleIJPFviiEiiEEEEEE to i8*), i8* bitcast (void (%"struct.std::thread::_State"*)* @_ZNSt6thread6_StateD2Ev to i8*), i8* bitcast (void (%"struct.std::thread::_State_impl"*)* @_ZNSt6thread11_State_implINS_8_InvokerISt5tupleIJPFviiEiiEEEEED0Ev to i8*), i8* bitcast (void (%"struct.std::thread::_State_impl"*)* @_ZNSt6thread11_State_implINS_8_InvokerISt5tupleIJPFviiEiiEEEEE6_M_runEv to i8*)] }, comdat, align 8
@_ZTVN10__cxxabiv120__si_class_type_infoE = external global i8*
@_ZTSNSt6thread11_State_implINS_8_InvokerISt5tupleIJPFviiEiiEEEEEE = linkonce_odr dso_local constant [62 x i8] c"NSt6thread11_State_implINS_8_InvokerISt5tupleIJPFviiEiiEEEEEE\00", comdat, align 1
@_ZTINSt6thread6_StateE = external constant i8*
@_ZTINSt6thread11_State_implINS_8_InvokerISt5tupleIJPFviiEiiEEEEEE = linkonce_odr dso_local constant { i8*, i8*, i8* } { i8* bitcast (i8** getelementptr inbounds (i8*, i8** @_ZTVN10__cxxabiv120__si_class_type_infoE, i64 2) to i8*), i8* getelementptr inbounds ([62 x i8], [62 x i8]* @_ZTSNSt6thread11_State_implINS_8_InvokerISt5tupleIJPFviiEiiEEEEEE, i32 0, i32 0), i8* bitcast (i8** @_ZTINSt6thread6_StateE to i8*) }, comdat, align 8
@.str.4 = private unnamed_addr constant [26 x i8] c"vector::_M_realloc_insert\00", align 1
@llvm.global_ctors = appending global [1 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 65535, void ()* @_GLOBAL__sub_I_parallelFor.cpp, i8* null }]

declare void @_ZNSt8ios_base4InitC1Ev(%"class.std::ios_base::Init"* noundef nonnull align 1 dereferenceable(1)) unnamed_addr #0

; Function Attrs: nounwind
declare void @_ZNSt8ios_base4InitD1Ev(%"class.std::ios_base::Init"* noundef nonnull align 1 dereferenceable(1)) unnamed_addr #1

; Function Attrs: nofree nounwind
declare i32 @__cxa_atexit(void (i8*)*, i8*, i8*) local_unnamed_addr #2

; Function Attrs: uwtable
define dso_local void @parallelFor(i32 noundef %0, i32 noundef %1, void (i32, i32)* noundef %2) local_unnamed_addr #3 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
  %4 = alloca %"class.std::unique_ptr", align 8
  %5 = alloca void (i32, i32)*, align 8
  %6 = alloca %"class.std::vector", align 8
  %7 = alloca i32, align 4
  %8 = alloca i32, align 4
  store void (i32, i32)* %2, void (i32, i32)** %5, align 8, !tbaa !5
  %9 = sub nsw i32 %1, %0
  %10 = sdiv i32 %9, 4
  %11 = srem i32 %9, 4
  %12 = bitcast %"class.std::vector"* %6 to i8*
  call void @llvm.lifetime.start.p0i8(i64 24, i8* nonnull %12) #14
  call void @llvm.memset.p0i8.i64(i8* noundef nonnull align 8 dereferenceable(24) %12, i8 0, i64 24, i1 false) #14
  %13 = bitcast i32* %7 to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* nonnull %13) #14
  store i32 %0, i32* %7, align 4, !tbaa !9
  %14 = bitcast i32* %8 to i8*
  %15 = getelementptr inbounds %"class.std::vector", %"class.std::vector"* %6, i64 0, i32 0, i32 0, i32 0, i32 1
  %16 = getelementptr inbounds %"class.std::vector", %"class.std::vector"* %6, i64 0, i32 0, i32 0, i32 0, i32 2
  %17 = bitcast %"class.std::unique_ptr"* %4 to i8*
  %18 = getelementptr inbounds %"class.std::unique_ptr", %"class.std::unique_ptr"* %4, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0
  call void @llvm.lifetime.start.p0i8(i64 4, i8* nonnull %14) #14
  %19 = icmp sgt i32 %11, 0
  %20 = zext i1 %19 to i32
  %21 = add nsw i32 %10, %20
  %22 = add i32 %21, %0
  store i32 %22, i32* %8, align 4, !tbaa !9
  %23 = invoke noundef nonnull align 8 dereferenceable(8) %"class.std::basic_ostream"* @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(%"class.std::basic_ostream"* noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, i8* noundef nonnull getelementptr inbounds ([18 x i8], [18 x i8]* @.str, i64 0, i64 0), i64 noundef 17)
          to label %24 unwind label %232

24:                                               ; preds = %3
  %25 = invoke noundef nonnull align 8 dereferenceable(8) %"class.std::basic_ostream"* @_ZNSolsEi(%"class.std::basic_ostream"* noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, i32 noundef 0)
          to label %26 unwind label %232

26:                                               ; preds = %24
  %27 = invoke noundef nonnull align 8 dereferenceable(8) %"class.std::basic_ostream"* @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(%"class.std::basic_ostream"* noundef nonnull align 8 dereferenceable(8) %25, i8* noundef nonnull getelementptr inbounds ([14 x i8], [14 x i8]* @.str.1, i64 0, i64 0), i64 noundef 13)
          to label %28 unwind label %232

28:                                               ; preds = %26
  %29 = load i32, i32* %7, align 4, !tbaa !9
  %30 = invoke noundef nonnull align 8 dereferenceable(8) %"class.std::basic_ostream"* @_ZNSolsEi(%"class.std::basic_ostream"* noundef nonnull align 8 dereferenceable(8) %25, i32 noundef %29)
          to label %31 unwind label %232

31:                                               ; preds = %28
  %32 = invoke noundef nonnull align 8 dereferenceable(8) %"class.std::basic_ostream"* @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(%"class.std::basic_ostream"* noundef nonnull align 8 dereferenceable(8) %30, i8* noundef nonnull getelementptr inbounds ([3 x i8], [3 x i8]* @.str.2, i64 0, i64 0), i64 noundef 2)
          to label %33 unwind label %232

33:                                               ; preds = %31
  %34 = load i32, i32* %8, align 4, !tbaa !9
  %35 = invoke noundef nonnull align 8 dereferenceable(8) %"class.std::basic_ostream"* @_ZNSolsEi(%"class.std::basic_ostream"* noundef nonnull align 8 dereferenceable(8) %30, i32 noundef %34)
          to label %36 unwind label %232

36:                                               ; preds = %33
  %37 = invoke noundef nonnull align 8 dereferenceable(8) %"class.std::basic_ostream"* @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(%"class.std::basic_ostream"* noundef nonnull align 8 dereferenceable(8) %35, i8* noundef nonnull getelementptr inbounds ([3 x i8], [3 x i8]* @.str.3, i64 0, i64 0), i64 noundef 2)
          to label %38 unwind label %232

38:                                               ; preds = %36
  %39 = load %"class.std::thread"*, %"class.std::thread"** %15, align 8, !tbaa !11
  %40 = load %"class.std::thread"*, %"class.std::thread"** %16, align 8, !tbaa !13
  %41 = icmp eq %"class.std::thread"* %39, %40
  br i1 %41, label %75, label %42

42:                                               ; preds = %38
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %17)
  %43 = getelementptr inbounds %"class.std::thread", %"class.std::thread"* %39, i64 0, i32 0, i32 0
  store i64 0, i64* %43, align 8, !tbaa !14
  %44 = invoke noalias noundef nonnull dereferenceable(24) i8* @_Znwm(i64 noundef 24) #15
          to label %45 unwind label %232

45:                                               ; preds = %42
  %46 = bitcast i8* %44 to %"struct.std::thread::_State_impl"*
  %47 = getelementptr inbounds %"struct.std::thread::_State_impl", %"struct.std::thread::_State_impl"* %46, i64 0, i32 0, i32 0
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [5 x i8*] }, { [5 x i8*] }* @_ZTVNSt6thread11_State_implINS_8_InvokerISt5tupleIJPFviiEiiEEEEEE, i64 0, inrange i32 0, i64 2) to i32 (...)**), i32 (...)*** %47, align 8, !tbaa !17
  %48 = getelementptr inbounds %"struct.std::thread::_State_impl", %"struct.std::thread::_State_impl"* %46, i64 0, i32 1, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0
  %49 = load i32, i32* %8, align 4, !tbaa !9
  store i32 %49, i32* %48, align 8, !tbaa !19
  %50 = getelementptr inbounds %"struct.std::thread::_State_impl", %"struct.std::thread::_State_impl"* %46, i64 0, i32 1, i32 0, i32 0, i32 0, i32 1, i32 0
  %51 = load i32, i32* %7, align 4, !tbaa !9
  store i32 %51, i32* %50, align 4, !tbaa !21
  %52 = getelementptr inbounds %"struct.std::thread::_State_impl", %"struct.std::thread::_State_impl"* %46, i64 0, i32 1, i32 0, i32 0, i32 1, i32 0
  %53 = load void (i32, i32)*, void (i32, i32)** %5, align 8, !tbaa !5
  store void (i32, i32)* %53, void (i32, i32)** %52, align 8, !tbaa !23
  %54 = getelementptr %"struct.std::thread::_State_impl", %"struct.std::thread::_State_impl"* %46, i64 0, i32 0
  store %"struct.std::thread::_State"* %54, %"struct.std::thread::_State"** %18, align 8, !tbaa !5
  invoke void @_ZNSt6thread15_M_start_threadESt10unique_ptrINS_6_StateESt14default_deleteIS1_EEPFvvE(%"class.std::thread"* noundef nonnull align 8 dereferenceable(8) %39, %"class.std::unique_ptr"* noundef nonnull %4, void ()* noundef null)
          to label %55 unwind label %63

55:                                               ; preds = %45
  %56 = load %"struct.std::thread::_State"*, %"struct.std::thread::_State"** %18, align 8, !tbaa !5
  %57 = icmp eq %"struct.std::thread::_State"* %56, null
  br i1 %57, label %72, label %58

58:                                               ; preds = %55
  %59 = bitcast %"struct.std::thread::_State"* %56 to void (%"struct.std::thread::_State"*)***
  %60 = load void (%"struct.std::thread::_State"*)**, void (%"struct.std::thread::_State"*)*** %59, align 8, !tbaa !17
  %61 = getelementptr inbounds void (%"struct.std::thread::_State"*)*, void (%"struct.std::thread::_State"*)** %60, i64 1
  %62 = load void (%"struct.std::thread::_State"*)*, void (%"struct.std::thread::_State"*)** %61, align 8
  call void %62(%"struct.std::thread::_State"* noundef nonnull align 8 dereferenceable(8) %56) #14
  br label %72

63:                                               ; preds = %201, %154, %104, %45
  %64 = landingpad { i8*, i32 }
          cleanup
  %65 = load %"struct.std::thread::_State"*, %"struct.std::thread::_State"** %18, align 8, !tbaa !5
  %66 = icmp eq %"struct.std::thread::_State"* %65, null
  br i1 %66, label %234, label %67

67:                                               ; preds = %63
  %68 = bitcast %"struct.std::thread::_State"* %65 to void (%"struct.std::thread::_State"*)***
  %69 = load void (%"struct.std::thread::_State"*)**, void (%"struct.std::thread::_State"*)*** %68, align 8, !tbaa !17
  %70 = getelementptr inbounds void (%"struct.std::thread::_State"*)*, void (%"struct.std::thread::_State"*)** %69, i64 1
  %71 = load void (%"struct.std::thread::_State"*)*, void (%"struct.std::thread::_State"*)** %70, align 8
  call void %71(%"struct.std::thread::_State"* noundef nonnull align 8 dereferenceable(8) %65) #14
  br label %234

72:                                               ; preds = %58, %55
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %17)
  %73 = load %"class.std::thread"*, %"class.std::thread"** %15, align 8, !tbaa !11
  %74 = getelementptr inbounds %"class.std::thread", %"class.std::thread"* %73, i64 1
  store %"class.std::thread"* %74, %"class.std::thread"** %15, align 8, !tbaa !11
  br label %76

75:                                               ; preds = %38
  invoke void @_ZNSt6vectorISt6threadSaIS0_EE17_M_realloc_insertIJRPFviiERiS7_EEEvN9__gnu_cxx17__normal_iteratorIPS0_S2_EEDpOT_(%"class.std::vector"* noundef nonnull align 8 dereferenceable(24) %6, %"class.std::thread"* %39, void (i32, i32)** noundef nonnull align 8 dereferenceable(8) %5, i32* noundef nonnull align 4 dereferenceable(4) %7, i32* noundef nonnull align 4 dereferenceable(4) %8)
          to label %76 unwind label %232

76:                                               ; preds = %72, %75
  %77 = load i32, i32* %8, align 4, !tbaa !9
  store i32 %77, i32* %7, align 4, !tbaa !9
  call void @llvm.lifetime.end.p0i8(i64 4, i8* nonnull %14) #14
  call void @llvm.lifetime.start.p0i8(i64 4, i8* nonnull %14) #14
  %78 = icmp sgt i32 %11, 1
  %79 = zext i1 %78 to i32
  %80 = add nsw i32 %10, %79
  %81 = add i32 %80, %77
  store i32 %81, i32* %8, align 4, !tbaa !9
  %82 = invoke noundef nonnull align 8 dereferenceable(8) %"class.std::basic_ostream"* @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(%"class.std::basic_ostream"* noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, i8* noundef nonnull getelementptr inbounds ([18 x i8], [18 x i8]* @.str, i64 0, i64 0), i64 noundef 17)
          to label %83 unwind label %232

83:                                               ; preds = %76
  %84 = invoke noundef nonnull align 8 dereferenceable(8) %"class.std::basic_ostream"* @_ZNSolsEi(%"class.std::basic_ostream"* noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, i32 noundef 1)
          to label %85 unwind label %232

85:                                               ; preds = %83
  %86 = invoke noundef nonnull align 8 dereferenceable(8) %"class.std::basic_ostream"* @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(%"class.std::basic_ostream"* noundef nonnull align 8 dereferenceable(8) %84, i8* noundef nonnull getelementptr inbounds ([14 x i8], [14 x i8]* @.str.1, i64 0, i64 0), i64 noundef 13)
          to label %87 unwind label %232

87:                                               ; preds = %85
  %88 = load i32, i32* %7, align 4, !tbaa !9
  %89 = invoke noundef nonnull align 8 dereferenceable(8) %"class.std::basic_ostream"* @_ZNSolsEi(%"class.std::basic_ostream"* noundef nonnull align 8 dereferenceable(8) %84, i32 noundef %88)
          to label %90 unwind label %232

90:                                               ; preds = %87
  %91 = invoke noundef nonnull align 8 dereferenceable(8) %"class.std::basic_ostream"* @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(%"class.std::basic_ostream"* noundef nonnull align 8 dereferenceable(8) %89, i8* noundef nonnull getelementptr inbounds ([3 x i8], [3 x i8]* @.str.2, i64 0, i64 0), i64 noundef 2)
          to label %92 unwind label %232

92:                                               ; preds = %90
  %93 = load i32, i32* %8, align 4, !tbaa !9
  %94 = invoke noundef nonnull align 8 dereferenceable(8) %"class.std::basic_ostream"* @_ZNSolsEi(%"class.std::basic_ostream"* noundef nonnull align 8 dereferenceable(8) %89, i32 noundef %93)
          to label %95 unwind label %232

95:                                               ; preds = %92
  %96 = invoke noundef nonnull align 8 dereferenceable(8) %"class.std::basic_ostream"* @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(%"class.std::basic_ostream"* noundef nonnull align 8 dereferenceable(8) %94, i8* noundef nonnull getelementptr inbounds ([3 x i8], [3 x i8]* @.str.3, i64 0, i64 0), i64 noundef 2)
          to label %97 unwind label %232

97:                                               ; preds = %95
  %98 = load %"class.std::thread"*, %"class.std::thread"** %15, align 8, !tbaa !11
  %99 = load %"class.std::thread"*, %"class.std::thread"** %16, align 8, !tbaa !13
  %100 = icmp eq %"class.std::thread"* %98, %99
  br i1 %100, label %125, label %101

101:                                              ; preds = %97
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %17)
  %102 = getelementptr inbounds %"class.std::thread", %"class.std::thread"* %98, i64 0, i32 0, i32 0
  store i64 0, i64* %102, align 8, !tbaa !14
  %103 = invoke noalias noundef nonnull dereferenceable(24) i8* @_Znwm(i64 noundef 24) #15
          to label %104 unwind label %232

104:                                              ; preds = %101
  %105 = bitcast i8* %103 to %"struct.std::thread::_State_impl"*
  %106 = getelementptr inbounds %"struct.std::thread::_State_impl", %"struct.std::thread::_State_impl"* %105, i64 0, i32 0, i32 0
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [5 x i8*] }, { [5 x i8*] }* @_ZTVNSt6thread11_State_implINS_8_InvokerISt5tupleIJPFviiEiiEEEEEE, i64 0, inrange i32 0, i64 2) to i32 (...)**), i32 (...)*** %106, align 8, !tbaa !17
  %107 = getelementptr inbounds %"struct.std::thread::_State_impl", %"struct.std::thread::_State_impl"* %105, i64 0, i32 1, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0
  %108 = load i32, i32* %8, align 4, !tbaa !9
  store i32 %108, i32* %107, align 8, !tbaa !19
  %109 = getelementptr inbounds %"struct.std::thread::_State_impl", %"struct.std::thread::_State_impl"* %105, i64 0, i32 1, i32 0, i32 0, i32 0, i32 1, i32 0
  %110 = load i32, i32* %7, align 4, !tbaa !9
  store i32 %110, i32* %109, align 4, !tbaa !21
  %111 = getelementptr inbounds %"struct.std::thread::_State_impl", %"struct.std::thread::_State_impl"* %105, i64 0, i32 1, i32 0, i32 0, i32 1, i32 0
  %112 = load void (i32, i32)*, void (i32, i32)** %5, align 8, !tbaa !5
  store void (i32, i32)* %112, void (i32, i32)** %111, align 8, !tbaa !23
  %113 = getelementptr %"struct.std::thread::_State_impl", %"struct.std::thread::_State_impl"* %105, i64 0, i32 0
  store %"struct.std::thread::_State"* %113, %"struct.std::thread::_State"** %18, align 8, !tbaa !5
  invoke void @_ZNSt6thread15_M_start_threadESt10unique_ptrINS_6_StateESt14default_deleteIS1_EEPFvvE(%"class.std::thread"* noundef nonnull align 8 dereferenceable(8) %98, %"class.std::unique_ptr"* noundef nonnull %4, void ()* noundef null)
          to label %114 unwind label %63

114:                                              ; preds = %104
  %115 = load %"struct.std::thread::_State"*, %"struct.std::thread::_State"** %18, align 8, !tbaa !5
  %116 = icmp eq %"struct.std::thread::_State"* %115, null
  br i1 %116, label %122, label %117

117:                                              ; preds = %114
  %118 = bitcast %"struct.std::thread::_State"* %115 to void (%"struct.std::thread::_State"*)***
  %119 = load void (%"struct.std::thread::_State"*)**, void (%"struct.std::thread::_State"*)*** %118, align 8, !tbaa !17
  %120 = getelementptr inbounds void (%"struct.std::thread::_State"*)*, void (%"struct.std::thread::_State"*)** %119, i64 1
  %121 = load void (%"struct.std::thread::_State"*)*, void (%"struct.std::thread::_State"*)** %120, align 8
  call void %121(%"struct.std::thread::_State"* noundef nonnull align 8 dereferenceable(8) %115) #14
  br label %122

122:                                              ; preds = %117, %114
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %17)
  %123 = load %"class.std::thread"*, %"class.std::thread"** %15, align 8, !tbaa !11
  %124 = getelementptr inbounds %"class.std::thread", %"class.std::thread"* %123, i64 1
  store %"class.std::thread"* %124, %"class.std::thread"** %15, align 8, !tbaa !11
  br label %126

125:                                              ; preds = %97
  invoke void @_ZNSt6vectorISt6threadSaIS0_EE17_M_realloc_insertIJRPFviiERiS7_EEEvN9__gnu_cxx17__normal_iteratorIPS0_S2_EEDpOT_(%"class.std::vector"* noundef nonnull align 8 dereferenceable(24) %6, %"class.std::thread"* %98, void (i32, i32)** noundef nonnull align 8 dereferenceable(8) %5, i32* noundef nonnull align 4 dereferenceable(4) %7, i32* noundef nonnull align 4 dereferenceable(4) %8)
          to label %126 unwind label %232

126:                                              ; preds = %125, %122
  %127 = load i32, i32* %8, align 4, !tbaa !9
  store i32 %127, i32* %7, align 4, !tbaa !9
  call void @llvm.lifetime.end.p0i8(i64 4, i8* nonnull %14) #14
  call void @llvm.lifetime.start.p0i8(i64 4, i8* nonnull %14) #14
  %128 = icmp sgt i32 %11, 2
  %129 = zext i1 %128 to i32
  %130 = add nsw i32 %10, %129
  %131 = add i32 %130, %127
  store i32 %131, i32* %8, align 4, !tbaa !9
  %132 = invoke noundef nonnull align 8 dereferenceable(8) %"class.std::basic_ostream"* @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(%"class.std::basic_ostream"* noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, i8* noundef nonnull getelementptr inbounds ([18 x i8], [18 x i8]* @.str, i64 0, i64 0), i64 noundef 17)
          to label %133 unwind label %232

133:                                              ; preds = %126
  %134 = invoke noundef nonnull align 8 dereferenceable(8) %"class.std::basic_ostream"* @_ZNSolsEi(%"class.std::basic_ostream"* noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, i32 noundef 2)
          to label %135 unwind label %232

135:                                              ; preds = %133
  %136 = invoke noundef nonnull align 8 dereferenceable(8) %"class.std::basic_ostream"* @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(%"class.std::basic_ostream"* noundef nonnull align 8 dereferenceable(8) %134, i8* noundef nonnull getelementptr inbounds ([14 x i8], [14 x i8]* @.str.1, i64 0, i64 0), i64 noundef 13)
          to label %137 unwind label %232

137:                                              ; preds = %135
  %138 = load i32, i32* %7, align 4, !tbaa !9
  %139 = invoke noundef nonnull align 8 dereferenceable(8) %"class.std::basic_ostream"* @_ZNSolsEi(%"class.std::basic_ostream"* noundef nonnull align 8 dereferenceable(8) %134, i32 noundef %138)
          to label %140 unwind label %232

140:                                              ; preds = %137
  %141 = invoke noundef nonnull align 8 dereferenceable(8) %"class.std::basic_ostream"* @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(%"class.std::basic_ostream"* noundef nonnull align 8 dereferenceable(8) %139, i8* noundef nonnull getelementptr inbounds ([3 x i8], [3 x i8]* @.str.2, i64 0, i64 0), i64 noundef 2)
          to label %142 unwind label %232

142:                                              ; preds = %140
  %143 = load i32, i32* %8, align 4, !tbaa !9
  %144 = invoke noundef nonnull align 8 dereferenceable(8) %"class.std::basic_ostream"* @_ZNSolsEi(%"class.std::basic_ostream"* noundef nonnull align 8 dereferenceable(8) %139, i32 noundef %143)
          to label %145 unwind label %232

145:                                              ; preds = %142
  %146 = invoke noundef nonnull align 8 dereferenceable(8) %"class.std::basic_ostream"* @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(%"class.std::basic_ostream"* noundef nonnull align 8 dereferenceable(8) %144, i8* noundef nonnull getelementptr inbounds ([3 x i8], [3 x i8]* @.str.3, i64 0, i64 0), i64 noundef 2)
          to label %147 unwind label %232

147:                                              ; preds = %145
  %148 = load %"class.std::thread"*, %"class.std::thread"** %15, align 8, !tbaa !11
  %149 = load %"class.std::thread"*, %"class.std::thread"** %16, align 8, !tbaa !13
  %150 = icmp eq %"class.std::thread"* %148, %149
  br i1 %150, label %175, label %151

151:                                              ; preds = %147
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %17)
  %152 = getelementptr inbounds %"class.std::thread", %"class.std::thread"* %148, i64 0, i32 0, i32 0
  store i64 0, i64* %152, align 8, !tbaa !14
  %153 = invoke noalias noundef nonnull dereferenceable(24) i8* @_Znwm(i64 noundef 24) #15
          to label %154 unwind label %232

154:                                              ; preds = %151
  %155 = bitcast i8* %153 to %"struct.std::thread::_State_impl"*
  %156 = getelementptr inbounds %"struct.std::thread::_State_impl", %"struct.std::thread::_State_impl"* %155, i64 0, i32 0, i32 0
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [5 x i8*] }, { [5 x i8*] }* @_ZTVNSt6thread11_State_implINS_8_InvokerISt5tupleIJPFviiEiiEEEEEE, i64 0, inrange i32 0, i64 2) to i32 (...)**), i32 (...)*** %156, align 8, !tbaa !17
  %157 = getelementptr inbounds %"struct.std::thread::_State_impl", %"struct.std::thread::_State_impl"* %155, i64 0, i32 1, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0
  %158 = load i32, i32* %8, align 4, !tbaa !9
  store i32 %158, i32* %157, align 8, !tbaa !19
  %159 = getelementptr inbounds %"struct.std::thread::_State_impl", %"struct.std::thread::_State_impl"* %155, i64 0, i32 1, i32 0, i32 0, i32 0, i32 1, i32 0
  %160 = load i32, i32* %7, align 4, !tbaa !9
  store i32 %160, i32* %159, align 4, !tbaa !21
  %161 = getelementptr inbounds %"struct.std::thread::_State_impl", %"struct.std::thread::_State_impl"* %155, i64 0, i32 1, i32 0, i32 0, i32 1, i32 0
  %162 = load void (i32, i32)*, void (i32, i32)** %5, align 8, !tbaa !5
  store void (i32, i32)* %162, void (i32, i32)** %161, align 8, !tbaa !23
  %163 = getelementptr %"struct.std::thread::_State_impl", %"struct.std::thread::_State_impl"* %155, i64 0, i32 0
  store %"struct.std::thread::_State"* %163, %"struct.std::thread::_State"** %18, align 8, !tbaa !5
  invoke void @_ZNSt6thread15_M_start_threadESt10unique_ptrINS_6_StateESt14default_deleteIS1_EEPFvvE(%"class.std::thread"* noundef nonnull align 8 dereferenceable(8) %148, %"class.std::unique_ptr"* noundef nonnull %4, void ()* noundef null)
          to label %164 unwind label %63

164:                                              ; preds = %154
  %165 = load %"struct.std::thread::_State"*, %"struct.std::thread::_State"** %18, align 8, !tbaa !5
  %166 = icmp eq %"struct.std::thread::_State"* %165, null
  br i1 %166, label %172, label %167

167:                                              ; preds = %164
  %168 = bitcast %"struct.std::thread::_State"* %165 to void (%"struct.std::thread::_State"*)***
  %169 = load void (%"struct.std::thread::_State"*)**, void (%"struct.std::thread::_State"*)*** %168, align 8, !tbaa !17
  %170 = getelementptr inbounds void (%"struct.std::thread::_State"*)*, void (%"struct.std::thread::_State"*)** %169, i64 1
  %171 = load void (%"struct.std::thread::_State"*)*, void (%"struct.std::thread::_State"*)** %170, align 8
  call void %171(%"struct.std::thread::_State"* noundef nonnull align 8 dereferenceable(8) %165) #14
  br label %172

172:                                              ; preds = %167, %164
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %17)
  %173 = load %"class.std::thread"*, %"class.std::thread"** %15, align 8, !tbaa !11
  %174 = getelementptr inbounds %"class.std::thread", %"class.std::thread"* %173, i64 1
  store %"class.std::thread"* %174, %"class.std::thread"** %15, align 8, !tbaa !11
  br label %176

175:                                              ; preds = %147
  invoke void @_ZNSt6vectorISt6threadSaIS0_EE17_M_realloc_insertIJRPFviiERiS7_EEEvN9__gnu_cxx17__normal_iteratorIPS0_S2_EEDpOT_(%"class.std::vector"* noundef nonnull align 8 dereferenceable(24) %6, %"class.std::thread"* %148, void (i32, i32)** noundef nonnull align 8 dereferenceable(8) %5, i32* noundef nonnull align 4 dereferenceable(4) %7, i32* noundef nonnull align 4 dereferenceable(4) %8)
          to label %176 unwind label %232

176:                                              ; preds = %175, %172
  %177 = load i32, i32* %8, align 4, !tbaa !9
  store i32 %177, i32* %7, align 4, !tbaa !9
  call void @llvm.lifetime.end.p0i8(i64 4, i8* nonnull %14) #14
  call void @llvm.lifetime.start.p0i8(i64 4, i8* nonnull %14) #14
  %178 = add i32 %10, %177
  store i32 %178, i32* %8, align 4, !tbaa !9
  %179 = invoke noundef nonnull align 8 dereferenceable(8) %"class.std::basic_ostream"* @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(%"class.std::basic_ostream"* noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, i8* noundef nonnull getelementptr inbounds ([18 x i8], [18 x i8]* @.str, i64 0, i64 0), i64 noundef 17)
          to label %180 unwind label %232

180:                                              ; preds = %176
  %181 = invoke noundef nonnull align 8 dereferenceable(8) %"class.std::basic_ostream"* @_ZNSolsEi(%"class.std::basic_ostream"* noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, i32 noundef 3)
          to label %182 unwind label %232

182:                                              ; preds = %180
  %183 = invoke noundef nonnull align 8 dereferenceable(8) %"class.std::basic_ostream"* @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(%"class.std::basic_ostream"* noundef nonnull align 8 dereferenceable(8) %181, i8* noundef nonnull getelementptr inbounds ([14 x i8], [14 x i8]* @.str.1, i64 0, i64 0), i64 noundef 13)
          to label %184 unwind label %232

184:                                              ; preds = %182
  %185 = load i32, i32* %7, align 4, !tbaa !9
  %186 = invoke noundef nonnull align 8 dereferenceable(8) %"class.std::basic_ostream"* @_ZNSolsEi(%"class.std::basic_ostream"* noundef nonnull align 8 dereferenceable(8) %181, i32 noundef %185)
          to label %187 unwind label %232

187:                                              ; preds = %184
  %188 = invoke noundef nonnull align 8 dereferenceable(8) %"class.std::basic_ostream"* @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(%"class.std::basic_ostream"* noundef nonnull align 8 dereferenceable(8) %186, i8* noundef nonnull getelementptr inbounds ([3 x i8], [3 x i8]* @.str.2, i64 0, i64 0), i64 noundef 2)
          to label %189 unwind label %232

189:                                              ; preds = %187
  %190 = load i32, i32* %8, align 4, !tbaa !9
  %191 = invoke noundef nonnull align 8 dereferenceable(8) %"class.std::basic_ostream"* @_ZNSolsEi(%"class.std::basic_ostream"* noundef nonnull align 8 dereferenceable(8) %186, i32 noundef %190)
          to label %192 unwind label %232

192:                                              ; preds = %189
  %193 = invoke noundef nonnull align 8 dereferenceable(8) %"class.std::basic_ostream"* @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(%"class.std::basic_ostream"* noundef nonnull align 8 dereferenceable(8) %191, i8* noundef nonnull getelementptr inbounds ([3 x i8], [3 x i8]* @.str.3, i64 0, i64 0), i64 noundef 2)
          to label %194 unwind label %232

194:                                              ; preds = %192
  %195 = load %"class.std::thread"*, %"class.std::thread"** %15, align 8, !tbaa !11
  %196 = load %"class.std::thread"*, %"class.std::thread"** %16, align 8, !tbaa !13
  %197 = icmp eq %"class.std::thread"* %195, %196
  br i1 %197, label %222, label %198

198:                                              ; preds = %194
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %17)
  %199 = getelementptr inbounds %"class.std::thread", %"class.std::thread"* %195, i64 0, i32 0, i32 0
  store i64 0, i64* %199, align 8, !tbaa !14
  %200 = invoke noalias noundef nonnull dereferenceable(24) i8* @_Znwm(i64 noundef 24) #15
          to label %201 unwind label %232

201:                                              ; preds = %198
  %202 = bitcast i8* %200 to %"struct.std::thread::_State_impl"*
  %203 = getelementptr inbounds %"struct.std::thread::_State_impl", %"struct.std::thread::_State_impl"* %202, i64 0, i32 0, i32 0
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [5 x i8*] }, { [5 x i8*] }* @_ZTVNSt6thread11_State_implINS_8_InvokerISt5tupleIJPFviiEiiEEEEEE, i64 0, inrange i32 0, i64 2) to i32 (...)**), i32 (...)*** %203, align 8, !tbaa !17
  %204 = getelementptr inbounds %"struct.std::thread::_State_impl", %"struct.std::thread::_State_impl"* %202, i64 0, i32 1, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0
  %205 = load i32, i32* %8, align 4, !tbaa !9
  store i32 %205, i32* %204, align 8, !tbaa !19
  %206 = getelementptr inbounds %"struct.std::thread::_State_impl", %"struct.std::thread::_State_impl"* %202, i64 0, i32 1, i32 0, i32 0, i32 0, i32 1, i32 0
  %207 = load i32, i32* %7, align 4, !tbaa !9
  store i32 %207, i32* %206, align 4, !tbaa !21
  %208 = getelementptr inbounds %"struct.std::thread::_State_impl", %"struct.std::thread::_State_impl"* %202, i64 0, i32 1, i32 0, i32 0, i32 1, i32 0
  %209 = load void (i32, i32)*, void (i32, i32)** %5, align 8, !tbaa !5
  store void (i32, i32)* %209, void (i32, i32)** %208, align 8, !tbaa !23
  %210 = getelementptr %"struct.std::thread::_State_impl", %"struct.std::thread::_State_impl"* %202, i64 0, i32 0
  store %"struct.std::thread::_State"* %210, %"struct.std::thread::_State"** %18, align 8, !tbaa !5
  invoke void @_ZNSt6thread15_M_start_threadESt10unique_ptrINS_6_StateESt14default_deleteIS1_EEPFvvE(%"class.std::thread"* noundef nonnull align 8 dereferenceable(8) %195, %"class.std::unique_ptr"* noundef nonnull %4, void ()* noundef null)
          to label %211 unwind label %63

211:                                              ; preds = %201
  %212 = load %"struct.std::thread::_State"*, %"struct.std::thread::_State"** %18, align 8, !tbaa !5
  %213 = icmp eq %"struct.std::thread::_State"* %212, null
  br i1 %213, label %219, label %214

214:                                              ; preds = %211
  %215 = bitcast %"struct.std::thread::_State"* %212 to void (%"struct.std::thread::_State"*)***
  %216 = load void (%"struct.std::thread::_State"*)**, void (%"struct.std::thread::_State"*)*** %215, align 8, !tbaa !17
  %217 = getelementptr inbounds void (%"struct.std::thread::_State"*)*, void (%"struct.std::thread::_State"*)** %216, i64 1
  %218 = load void (%"struct.std::thread::_State"*)*, void (%"struct.std::thread::_State"*)** %217, align 8
  call void %218(%"struct.std::thread::_State"* noundef nonnull align 8 dereferenceable(8) %212) #14
  br label %219

219:                                              ; preds = %214, %211
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %17)
  %220 = load %"class.std::thread"*, %"class.std::thread"** %15, align 8, !tbaa !11
  %221 = getelementptr inbounds %"class.std::thread", %"class.std::thread"* %220, i64 1
  store %"class.std::thread"* %221, %"class.std::thread"** %15, align 8, !tbaa !11
  br label %225

222:                                              ; preds = %194
  invoke void @_ZNSt6vectorISt6threadSaIS0_EE17_M_realloc_insertIJRPFviiERiS7_EEEvN9__gnu_cxx17__normal_iteratorIPS0_S2_EEDpOT_(%"class.std::vector"* noundef nonnull align 8 dereferenceable(24) %6, %"class.std::thread"* %195, void (i32, i32)** noundef nonnull align 8 dereferenceable(8) %5, i32* noundef nonnull align 4 dereferenceable(4) %7, i32* noundef nonnull align 4 dereferenceable(4) %8)
          to label %223 unwind label %232

223:                                              ; preds = %222
  %224 = load %"class.std::thread"*, %"class.std::thread"** %15, align 8, !tbaa !5
  br label %225

225:                                              ; preds = %223, %219
  %226 = phi %"class.std::thread"* [ %224, %223 ], [ %221, %219 ]
  %227 = load i32, i32* %8, align 4, !tbaa !9
  store i32 %227, i32* %7, align 4, !tbaa !9
  call void @llvm.lifetime.end.p0i8(i64 4, i8* nonnull %14) #14
  %228 = getelementptr inbounds %"class.std::vector", %"class.std::vector"* %6, i64 0, i32 0, i32 0, i32 0, i32 0
  %229 = load %"class.std::thread"*, %"class.std::thread"** %228, align 8, !tbaa !5
  %230 = icmp eq %"class.std::thread"* %229, %226
  br i1 %230, label %231, label %255

231:                                              ; preds = %225
  call void @llvm.lifetime.end.p0i8(i64 4, i8* nonnull %13) #14
  br label %249

232:                                              ; preds = %222, %198, %192, %189, %187, %184, %182, %180, %176, %175, %151, %145, %142, %140, %137, %135, %133, %126, %125, %101, %95, %92, %90, %87, %85, %83, %76, %75, %42, %36, %31, %26, %3, %33, %28, %24
  %233 = landingpad { i8*, i32 }
          cleanup
  br label %234

234:                                              ; preds = %63, %67, %232
  %235 = phi { i8*, i32 } [ %233, %232 ], [ %64, %67 ], [ %64, %63 ]
  call void @llvm.lifetime.end.p0i8(i64 4, i8* nonnull %14) #14
  br label %262

236:                                              ; preds = %257
  %237 = load %"class.std::thread"*, %"class.std::thread"** %228, align 8, !tbaa !25
  %238 = load %"class.std::thread"*, %"class.std::thread"** %15, align 8, !tbaa !11
  call void @llvm.lifetime.end.p0i8(i64 4, i8* nonnull %13) #14
  %239 = icmp eq %"class.std::thread"* %237, %238
  br i1 %239, label %249, label %242

240:                                              ; preds = %242
  %241 = icmp eq %"class.std::thread"* %247, %238
  br i1 %241, label %249, label %242, !llvm.loop !26

242:                                              ; preds = %236, %240
  %243 = phi %"class.std::thread"* [ %247, %240 ], [ %237, %236 ]
  %244 = getelementptr inbounds %"class.std::thread", %"class.std::thread"* %243, i64 0, i32 0, i32 0
  %245 = load i64, i64* %244, align 8, !tbaa.struct !28
  %246 = icmp eq i64 %245, 0
  %247 = getelementptr inbounds %"class.std::thread", %"class.std::thread"* %243, i64 1
  br i1 %246, label %240, label %248

248:                                              ; preds = %242
  call void @_ZSt9terminatev() #16
  unreachable

249:                                              ; preds = %240, %231, %236
  %250 = phi %"class.std::thread"* [ %226, %231 ], [ %237, %236 ], [ %237, %240 ]
  %251 = icmp eq %"class.std::thread"* %250, null
  br i1 %251, label %254, label %252

252:                                              ; preds = %249
  %253 = bitcast %"class.std::thread"* %250 to i8*
  call void @_ZdlPv(i8* noundef %253) #17
  br label %254

254:                                              ; preds = %249, %252
  call void @llvm.lifetime.end.p0i8(i64 24, i8* nonnull %12) #14
  ret void

255:                                              ; preds = %225, %257
  %256 = phi %"class.std::thread"* [ %258, %257 ], [ %229, %225 ]
  invoke void @_ZNSt6thread4joinEv(%"class.std::thread"* noundef nonnull align 8 dereferenceable(8) %256)
          to label %257 unwind label %260

257:                                              ; preds = %255
  %258 = getelementptr inbounds %"class.std::thread", %"class.std::thread"* %256, i64 1
  %259 = icmp eq %"class.std::thread"* %258, %226
  br i1 %259, label %236, label %255

260:                                              ; preds = %255
  %261 = landingpad { i8*, i32 }
          cleanup
  br label %262

262:                                              ; preds = %260, %234
  %263 = phi { i8*, i32 } [ %235, %234 ], [ %261, %260 ]
  call void @llvm.lifetime.end.p0i8(i64 4, i8* nonnull %13) #14
  call void @_ZNSt6vectorISt6threadSaIS0_EED2Ev(%"class.std::vector"* noundef nonnull align 8 dereferenceable(24) %6) #14
  call void @llvm.lifetime.end.p0i8(i64 24, i8* nonnull %12) #14
  resume { i8*, i32 } %263
}

; Function Attrs: argmemonly mustprogress nofree nosync nounwind willreturn
declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture) #4

declare i32 @__gxx_personality_v0(...)

declare noundef nonnull align 8 dereferenceable(8) %"class.std::basic_ostream"* @_ZNSolsEi(%"class.std::basic_ostream"* noundef nonnull align 8 dereferenceable(8), i32 noundef) local_unnamed_addr #0

; Function Attrs: argmemonly mustprogress nofree nosync nounwind willreturn
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture) #4

declare void @_ZNSt6thread4joinEv(%"class.std::thread"* noundef nonnull align 8 dereferenceable(8)) local_unnamed_addr #0

; Function Attrs: nounwind uwtable
define linkonce_odr dso_local void @_ZNSt6vectorISt6threadSaIS0_EED2Ev(%"class.std::vector"* noundef nonnull align 8 dereferenceable(24) %0) unnamed_addr #5 comdat align 2 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
  %2 = getelementptr inbounds %"class.std::vector", %"class.std::vector"* %0, i64 0, i32 0, i32 0, i32 0, i32 0
  %3 = load %"class.std::thread"*, %"class.std::thread"** %2, align 8, !tbaa !25
  %4 = getelementptr inbounds %"class.std::vector", %"class.std::vector"* %0, i64 0, i32 0, i32 0, i32 0, i32 1
  %5 = load %"class.std::thread"*, %"class.std::thread"** %4, align 8, !tbaa !11
  %6 = icmp eq %"class.std::thread"* %3, %5
  br i1 %6, label %16, label %9

7:                                                ; preds = %9
  %8 = icmp eq %"class.std::thread"* %14, %5
  br i1 %8, label %16, label %9, !llvm.loop !26

9:                                                ; preds = %1, %7
  %10 = phi %"class.std::thread"* [ %14, %7 ], [ %3, %1 ]
  %11 = getelementptr inbounds %"class.std::thread", %"class.std::thread"* %10, i64 0, i32 0, i32 0
  %12 = load i64, i64* %11, align 8, !tbaa.struct !28
  %13 = icmp eq i64 %12, 0
  %14 = getelementptr inbounds %"class.std::thread", %"class.std::thread"* %10, i64 1
  br i1 %13, label %7, label %15

15:                                               ; preds = %9
  tail call void @_ZSt9terminatev() #16
  unreachable

16:                                               ; preds = %7, %1
  %17 = icmp eq %"class.std::thread"* %3, null
  br i1 %17, label %20, label %18

18:                                               ; preds = %16
  %19 = bitcast %"class.std::thread"* %3 to i8*
  tail call void @_ZdlPv(i8* noundef %19) #17
  br label %20

20:                                               ; preds = %16, %18
  ret void
}

; Function Attrs: noinline noreturn nounwind
define linkonce_odr hidden void @__clang_call_terminate(i8* %0) local_unnamed_addr #6 comdat {
  %2 = tail call i8* @__cxa_begin_catch(i8* %0) #14
  tail call void @_ZSt9terminatev() #16
  unreachable
}

declare i8* @__cxa_begin_catch(i8*) local_unnamed_addr

declare void @_ZSt9terminatev() local_unnamed_addr

; Function Attrs: nobuiltin nounwind
declare void @_ZdlPv(i8* noundef) local_unnamed_addr #7

declare noundef nonnull align 8 dereferenceable(8) %"class.std::basic_ostream"* @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(%"class.std::basic_ostream"* noundef nonnull align 8 dereferenceable(8), i8* noundef, i64 noundef) local_unnamed_addr #0

; Function Attrs: uwtable
define linkonce_odr dso_local void @_ZNSt6vectorISt6threadSaIS0_EE17_M_realloc_insertIJRPFviiERiS7_EEEvN9__gnu_cxx17__normal_iteratorIPS0_S2_EEDpOT_(%"class.std::vector"* noundef nonnull align 8 dereferenceable(24) %0, %"class.std::thread"* %1, void (i32, i32)** noundef nonnull align 8 dereferenceable(8) %2, i32* noundef nonnull align 4 dereferenceable(4) %3, i32* noundef nonnull align 4 dereferenceable(4) %4) local_unnamed_addr #3 comdat align 2 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
  %6 = ptrtoint %"class.std::thread"* %1 to i64
  %7 = alloca %"class.std::unique_ptr", align 8
  %8 = getelementptr inbounds %"class.std::vector", %"class.std::vector"* %0, i64 0, i32 0, i32 0, i32 0, i32 1
  %9 = load %"class.std::thread"*, %"class.std::thread"** %8, align 8, !tbaa !11
  %10 = ptrtoint %"class.std::thread"* %9 to i64
  %11 = getelementptr inbounds %"class.std::vector", %"class.std::vector"* %0, i64 0, i32 0, i32 0, i32 0, i32 0
  %12 = load %"class.std::thread"*, %"class.std::thread"** %11, align 8, !tbaa !25
  %13 = ptrtoint %"class.std::thread"* %12 to i64
  %14 = ptrtoint %"class.std::thread"* %9 to i64
  %15 = ptrtoint %"class.std::thread"* %12 to i64
  %16 = sub i64 %14, %15
  %17 = ashr exact i64 %16, 3
  %18 = icmp eq i64 %16, 9223372036854775800
  br i1 %18, label %19, label %20

19:                                               ; preds = %5
  tail call void @_ZSt20__throw_length_errorPKc(i8* noundef getelementptr inbounds ([26 x i8], [26 x i8]* @.str.4, i64 0, i64 0)) #18
  unreachable

20:                                               ; preds = %5
  %21 = icmp eq i64 %16, 0
  %22 = select i1 %21, i64 1, i64 %17
  %23 = add nsw i64 %22, %17
  %24 = icmp ult i64 %23, %17
  %25 = icmp ugt i64 %23, 1152921504606846975
  %26 = or i1 %24, %25
  %27 = select i1 %26, i64 1152921504606846975, i64 %23
  %28 = ptrtoint %"class.std::thread"* %1 to i64
  %29 = sub i64 %28, %15
  %30 = ashr exact i64 %29, 3
  %31 = icmp eq i64 %27, 0
  br i1 %31, label %36, label %32

32:                                               ; preds = %20
  %33 = shl nuw nsw i64 %27, 3
  %34 = tail call noalias noundef nonnull i8* @_Znwm(i64 noundef %33) #15
  %35 = bitcast i8* %34 to %"class.std::thread"*
  br label %36

36:                                               ; preds = %20, %32
  %37 = phi %"class.std::thread"* [ %35, %32 ], [ null, %20 ]
  %38 = getelementptr inbounds %"class.std::thread", %"class.std::thread"* %37, i64 %30
  %39 = bitcast %"class.std::unique_ptr"* %7 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %39)
  %40 = getelementptr inbounds %"class.std::thread", %"class.std::thread"* %38, i64 0, i32 0, i32 0
  store i64 0, i64* %40, align 8, !tbaa !14
  %41 = invoke noalias noundef nonnull dereferenceable(24) i8* @_Znwm(i64 noundef 24) #15
          to label %42 unwind label %240

42:                                               ; preds = %36
  %43 = bitcast i8* %41 to %"struct.std::thread::_State_impl"*
  %44 = getelementptr inbounds %"struct.std::thread::_State_impl", %"struct.std::thread::_State_impl"* %43, i64 0, i32 0, i32 0
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [5 x i8*] }, { [5 x i8*] }* @_ZTVNSt6thread11_State_implINS_8_InvokerISt5tupleIJPFviiEiiEEEEEE, i64 0, inrange i32 0, i64 2) to i32 (...)**), i32 (...)*** %44, align 8, !tbaa !17
  %45 = getelementptr inbounds %"struct.std::thread::_State_impl", %"struct.std::thread::_State_impl"* %43, i64 0, i32 1, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0
  %46 = load i32, i32* %4, align 4, !tbaa !9
  store i32 %46, i32* %45, align 8, !tbaa !19
  %47 = getelementptr inbounds %"struct.std::thread::_State_impl", %"struct.std::thread::_State_impl"* %43, i64 0, i32 1, i32 0, i32 0, i32 0, i32 1, i32 0
  %48 = load i32, i32* %3, align 4, !tbaa !9
  store i32 %48, i32* %47, align 4, !tbaa !21
  %49 = getelementptr inbounds %"struct.std::thread::_State_impl", %"struct.std::thread::_State_impl"* %43, i64 0, i32 1, i32 0, i32 0, i32 1, i32 0
  %50 = load void (i32, i32)*, void (i32, i32)** %2, align 8, !tbaa !5
  store void (i32, i32)* %50, void (i32, i32)** %49, align 8, !tbaa !23
  %51 = getelementptr %"struct.std::thread::_State_impl", %"struct.std::thread::_State_impl"* %43, i64 0, i32 0
  %52 = getelementptr inbounds %"class.std::unique_ptr", %"class.std::unique_ptr"* %7, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0
  store %"struct.std::thread::_State"* %51, %"struct.std::thread::_State"** %52, align 8, !tbaa !5
  invoke void @_ZNSt6thread15_M_start_threadESt10unique_ptrINS_6_StateESt14default_deleteIS1_EEPFvvE(%"class.std::thread"* noundef nonnull align 8 dereferenceable(8) %38, %"class.std::unique_ptr"* noundef nonnull %7, void ()* noundef null)
          to label %53 unwind label %61

53:                                               ; preds = %42
  %54 = load %"struct.std::thread::_State"*, %"struct.std::thread::_State"** %52, align 8, !tbaa !5
  %55 = icmp eq %"struct.std::thread::_State"* %54, null
  br i1 %55, label %70, label %56

56:                                               ; preds = %53
  %57 = bitcast %"struct.std::thread::_State"* %54 to void (%"struct.std::thread::_State"*)***
  %58 = load void (%"struct.std::thread::_State"*)**, void (%"struct.std::thread::_State"*)*** %57, align 8, !tbaa !17
  %59 = getelementptr inbounds void (%"struct.std::thread::_State"*)*, void (%"struct.std::thread::_State"*)** %58, i64 1
  %60 = load void (%"struct.std::thread::_State"*)*, void (%"struct.std::thread::_State"*)** %59, align 8
  call void %60(%"struct.std::thread::_State"* noundef nonnull align 8 dereferenceable(8) %54) #14
  br label %70

61:                                               ; preds = %42
  %62 = landingpad { i8*, i32 }
          catch i8* null
  %63 = load %"struct.std::thread::_State"*, %"struct.std::thread::_State"** %52, align 8, !tbaa !5
  %64 = icmp eq %"struct.std::thread::_State"* %63, null
  br i1 %64, label %244, label %65

65:                                               ; preds = %61
  %66 = bitcast %"struct.std::thread::_State"* %63 to void (%"struct.std::thread::_State"*)***
  %67 = load void (%"struct.std::thread::_State"*)**, void (%"struct.std::thread::_State"*)*** %66, align 8, !tbaa !17
  %68 = getelementptr inbounds void (%"struct.std::thread::_State"*)*, void (%"struct.std::thread::_State"*)** %67, i64 1
  %69 = load void (%"struct.std::thread::_State"*)*, void (%"struct.std::thread::_State"*)** %68, align 8
  call void %69(%"struct.std::thread::_State"* noundef nonnull align 8 dereferenceable(8) %63) #14
  br label %244

70:                                               ; preds = %56, %53
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %39)
  %71 = icmp eq %"class.std::thread"* %12, %1
  br i1 %71, label %150, label %72

72:                                               ; preds = %70
  %73 = add i64 %6, -8
  %74 = sub i64 %73, %13
  %75 = lshr i64 %74, 3
  %76 = add nuw nsw i64 %75, 1
  %77 = icmp ult i64 %74, 24
  br i1 %77, label %138, label %78

78:                                               ; preds = %72
  %79 = and i64 %76, 4611686018427387900
  %80 = getelementptr %"class.std::thread", %"class.std::thread"* %37, i64 %79
  %81 = getelementptr %"class.std::thread", %"class.std::thread"* %12, i64 %79
  %82 = add nsw i64 %79, -4
  %83 = lshr exact i64 %82, 2
  %84 = add nuw nsw i64 %83, 1
  %85 = and i64 %84, 1
  %86 = icmp eq i64 %82, 0
  br i1 %86, label %120, label %87

87:                                               ; preds = %78
  %88 = and i64 %84, 9223372036854775806
  br label %89

89:                                               ; preds = %89, %87
  %90 = phi i64 [ 0, %87 ], [ %117, %89 ]
  %91 = phi i64 [ 0, %87 ], [ %118, %89 ]
  call void @llvm.experimental.noalias.scope.decl(metadata !30) #14
  call void @llvm.experimental.noalias.scope.decl(metadata !33) #14
  %92 = getelementptr %"class.std::thread", %"class.std::thread"* %37, i64 %90, i32 0, i32 0
  %93 = getelementptr %"class.std::thread", %"class.std::thread"* %12, i64 %90, i32 0, i32 0
  %94 = bitcast i64* %93 to <2 x i64>*
  %95 = load <2 x i64>, <2 x i64>* %94, align 8, !tbaa !29, !alias.scope !33, !noalias !30
  %96 = getelementptr i64, i64* %93, i64 2
  %97 = bitcast i64* %96 to <2 x i64>*
  %98 = load <2 x i64>, <2 x i64>* %97, align 8, !tbaa !29, !alias.scope !33, !noalias !30
  %99 = bitcast i64* %92 to <2 x i64>*
  store <2 x i64> %95, <2 x i64>* %99, align 8, !tbaa !29, !alias.scope !30, !noalias !33
  %100 = getelementptr i64, i64* %92, i64 2
  %101 = bitcast i64* %100 to <2 x i64>*
  store <2 x i64> %98, <2 x i64>* %101, align 8, !tbaa !29, !alias.scope !30, !noalias !33
  %102 = bitcast i64* %93 to <2 x i64>*
  store <2 x i64> zeroinitializer, <2 x i64>* %102, align 8, !tbaa !29, !alias.scope !33, !noalias !30
  %103 = bitcast i64* %96 to <2 x i64>*
  store <2 x i64> zeroinitializer, <2 x i64>* %103, align 8, !tbaa !29, !alias.scope !33, !noalias !30
  %104 = or i64 %90, 4
  call void @llvm.experimental.noalias.scope.decl(metadata !35) #14
  call void @llvm.experimental.noalias.scope.decl(metadata !37) #14
  %105 = getelementptr %"class.std::thread", %"class.std::thread"* %37, i64 %104, i32 0, i32 0
  %106 = getelementptr %"class.std::thread", %"class.std::thread"* %12, i64 %104, i32 0, i32 0
  %107 = bitcast i64* %106 to <2 x i64>*
  %108 = load <2 x i64>, <2 x i64>* %107, align 8, !tbaa !29, !alias.scope !37, !noalias !35
  %109 = getelementptr i64, i64* %106, i64 2
  %110 = bitcast i64* %109 to <2 x i64>*
  %111 = load <2 x i64>, <2 x i64>* %110, align 8, !tbaa !29, !alias.scope !37, !noalias !35
  %112 = bitcast i64* %105 to <2 x i64>*
  store <2 x i64> %108, <2 x i64>* %112, align 8, !tbaa !29, !alias.scope !35, !noalias !37
  %113 = getelementptr i64, i64* %105, i64 2
  %114 = bitcast i64* %113 to <2 x i64>*
  store <2 x i64> %111, <2 x i64>* %114, align 8, !tbaa !29, !alias.scope !35, !noalias !37
  %115 = bitcast i64* %106 to <2 x i64>*
  store <2 x i64> zeroinitializer, <2 x i64>* %115, align 8, !tbaa !29, !alias.scope !37, !noalias !35
  %116 = bitcast i64* %109 to <2 x i64>*
  store <2 x i64> zeroinitializer, <2 x i64>* %116, align 8, !tbaa !29, !alias.scope !37, !noalias !35
  %117 = add nuw i64 %90, 8
  %118 = add i64 %91, 2
  %119 = icmp eq i64 %118, %88
  br i1 %119, label %120, label %89, !llvm.loop !39

120:                                              ; preds = %89, %78
  %121 = phi i64 [ 0, %78 ], [ %117, %89 ]
  %122 = icmp eq i64 %85, 0
  br i1 %122, label %136, label %123

123:                                              ; preds = %120
  call void @llvm.experimental.noalias.scope.decl(metadata !30) #14
  call void @llvm.experimental.noalias.scope.decl(metadata !33) #14
  %124 = getelementptr %"class.std::thread", %"class.std::thread"* %37, i64 %121, i32 0, i32 0
  %125 = getelementptr %"class.std::thread", %"class.std::thread"* %12, i64 %121, i32 0, i32 0
  %126 = bitcast i64* %125 to <2 x i64>*
  %127 = load <2 x i64>, <2 x i64>* %126, align 8, !tbaa !29, !alias.scope !33, !noalias !30
  %128 = getelementptr i64, i64* %125, i64 2
  %129 = bitcast i64* %128 to <2 x i64>*
  %130 = load <2 x i64>, <2 x i64>* %129, align 8, !tbaa !29, !alias.scope !33, !noalias !30
  %131 = bitcast i64* %124 to <2 x i64>*
  store <2 x i64> %127, <2 x i64>* %131, align 8, !tbaa !29, !alias.scope !30, !noalias !33
  %132 = getelementptr i64, i64* %124, i64 2
  %133 = bitcast i64* %132 to <2 x i64>*
  store <2 x i64> %130, <2 x i64>* %133, align 8, !tbaa !29, !alias.scope !30, !noalias !33
  %134 = bitcast i64* %125 to <2 x i64>*
  store <2 x i64> zeroinitializer, <2 x i64>* %134, align 8, !tbaa !29, !alias.scope !33, !noalias !30
  %135 = bitcast i64* %128 to <2 x i64>*
  store <2 x i64> zeroinitializer, <2 x i64>* %135, align 8, !tbaa !29, !alias.scope !33, !noalias !30
  br label %136

136:                                              ; preds = %120, %123
  %137 = icmp eq i64 %76, %79
  br i1 %137, label %150, label %138

138:                                              ; preds = %72, %136
  %139 = phi %"class.std::thread"* [ %37, %72 ], [ %80, %136 ]
  %140 = phi %"class.std::thread"* [ %12, %72 ], [ %81, %136 ]
  br label %141

141:                                              ; preds = %138, %141
  %142 = phi %"class.std::thread"* [ %148, %141 ], [ %139, %138 ]
  %143 = phi %"class.std::thread"* [ %147, %141 ], [ %140, %138 ]
  call void @llvm.experimental.noalias.scope.decl(metadata !30) #14
  call void @llvm.experimental.noalias.scope.decl(metadata !33) #14
  %144 = getelementptr inbounds %"class.std::thread", %"class.std::thread"* %142, i64 0, i32 0, i32 0
  %145 = getelementptr inbounds %"class.std::thread", %"class.std::thread"* %143, i64 0, i32 0, i32 0
  %146 = load i64, i64* %145, align 8, !tbaa !29, !alias.scope !33, !noalias !30
  store i64 %146, i64* %144, align 8, !tbaa !29, !alias.scope !30, !noalias !33
  store i64 0, i64* %145, align 8, !tbaa !29, !alias.scope !33, !noalias !30
  %147 = getelementptr inbounds %"class.std::thread", %"class.std::thread"* %143, i64 1
  %148 = getelementptr inbounds %"class.std::thread", %"class.std::thread"* %142, i64 1
  %149 = icmp eq %"class.std::thread"* %147, %1
  br i1 %149, label %150, label %141, !llvm.loop !41

150:                                              ; preds = %141, %136, %70
  %151 = phi %"class.std::thread"* [ %37, %70 ], [ %80, %136 ], [ %148, %141 ]
  %152 = getelementptr %"class.std::thread", %"class.std::thread"* %151, i64 1
  %153 = icmp eq %"class.std::thread"* %9, %1
  br i1 %153, label %232, label %154

154:                                              ; preds = %150
  %155 = add i64 %10, -8
  %156 = sub i64 %155, %6
  %157 = lshr i64 %156, 3
  %158 = add nuw nsw i64 %157, 1
  %159 = icmp ult i64 %156, 24
  br i1 %159, label %220, label %160

160:                                              ; preds = %154
  %161 = and i64 %158, 4611686018427387900
  %162 = getelementptr %"class.std::thread", %"class.std::thread"* %152, i64 %161
  %163 = getelementptr %"class.std::thread", %"class.std::thread"* %1, i64 %161
  %164 = add nsw i64 %161, -4
  %165 = lshr exact i64 %164, 2
  %166 = add nuw nsw i64 %165, 1
  %167 = and i64 %166, 1
  %168 = icmp eq i64 %164, 0
  br i1 %168, label %202, label %169

169:                                              ; preds = %160
  %170 = and i64 %166, 9223372036854775806
  br label %171

171:                                              ; preds = %171, %169
  %172 = phi i64 [ 0, %169 ], [ %199, %171 ]
  %173 = phi i64 [ 0, %169 ], [ %200, %171 ]
  %174 = getelementptr %"class.std::thread", %"class.std::thread"* %152, i64 %172
  call void @llvm.experimental.noalias.scope.decl(metadata !43) #14
  call void @llvm.experimental.noalias.scope.decl(metadata !46) #14
  %175 = getelementptr %"class.std::thread", %"class.std::thread"* %1, i64 %172, i32 0, i32 0
  %176 = bitcast i64* %175 to <2 x i64>*
  %177 = load <2 x i64>, <2 x i64>* %176, align 8, !tbaa !29, !alias.scope !46, !noalias !43
  %178 = getelementptr i64, i64* %175, i64 2
  %179 = bitcast i64* %178 to <2 x i64>*
  %180 = load <2 x i64>, <2 x i64>* %179, align 8, !tbaa !29, !alias.scope !46, !noalias !43
  %181 = bitcast %"class.std::thread"* %174 to <2 x i64>*
  store <2 x i64> %177, <2 x i64>* %181, align 8, !tbaa !29, !alias.scope !43, !noalias !46
  %182 = getelementptr %"class.std::thread", %"class.std::thread"* %174, i64 2
  %183 = bitcast %"class.std::thread"* %182 to <2 x i64>*
  store <2 x i64> %180, <2 x i64>* %183, align 8, !tbaa !29, !alias.scope !43, !noalias !46
  %184 = bitcast i64* %175 to <2 x i64>*
  store <2 x i64> zeroinitializer, <2 x i64>* %184, align 8, !tbaa !29, !alias.scope !46, !noalias !43
  %185 = bitcast i64* %178 to <2 x i64>*
  store <2 x i64> zeroinitializer, <2 x i64>* %185, align 8, !tbaa !29, !alias.scope !46, !noalias !43
  %186 = or i64 %172, 4
  %187 = getelementptr %"class.std::thread", %"class.std::thread"* %152, i64 %186
  call void @llvm.experimental.noalias.scope.decl(metadata !48) #14
  call void @llvm.experimental.noalias.scope.decl(metadata !50) #14
  %188 = getelementptr %"class.std::thread", %"class.std::thread"* %1, i64 %186, i32 0, i32 0
  %189 = bitcast i64* %188 to <2 x i64>*
  %190 = load <2 x i64>, <2 x i64>* %189, align 8, !tbaa !29, !alias.scope !50, !noalias !48
  %191 = getelementptr i64, i64* %188, i64 2
  %192 = bitcast i64* %191 to <2 x i64>*
  %193 = load <2 x i64>, <2 x i64>* %192, align 8, !tbaa !29, !alias.scope !50, !noalias !48
  %194 = bitcast %"class.std::thread"* %187 to <2 x i64>*
  store <2 x i64> %190, <2 x i64>* %194, align 8, !tbaa !29, !alias.scope !48, !noalias !50
  %195 = getelementptr %"class.std::thread", %"class.std::thread"* %187, i64 2
  %196 = bitcast %"class.std::thread"* %195 to <2 x i64>*
  store <2 x i64> %193, <2 x i64>* %196, align 8, !tbaa !29, !alias.scope !48, !noalias !50
  %197 = bitcast i64* %188 to <2 x i64>*
  store <2 x i64> zeroinitializer, <2 x i64>* %197, align 8, !tbaa !29, !alias.scope !50, !noalias !48
  %198 = bitcast i64* %191 to <2 x i64>*
  store <2 x i64> zeroinitializer, <2 x i64>* %198, align 8, !tbaa !29, !alias.scope !50, !noalias !48
  %199 = add nuw i64 %172, 8
  %200 = add i64 %173, 2
  %201 = icmp eq i64 %200, %170
  br i1 %201, label %202, label %171, !llvm.loop !52

202:                                              ; preds = %171, %160
  %203 = phi i64 [ 0, %160 ], [ %199, %171 ]
  %204 = icmp eq i64 %167, 0
  br i1 %204, label %218, label %205

205:                                              ; preds = %202
  %206 = getelementptr %"class.std::thread", %"class.std::thread"* %152, i64 %203
  call void @llvm.experimental.noalias.scope.decl(metadata !43) #14
  call void @llvm.experimental.noalias.scope.decl(metadata !46) #14
  %207 = getelementptr %"class.std::thread", %"class.std::thread"* %1, i64 %203, i32 0, i32 0
  %208 = bitcast i64* %207 to <2 x i64>*
  %209 = load <2 x i64>, <2 x i64>* %208, align 8, !tbaa !29, !alias.scope !46, !noalias !43
  %210 = getelementptr i64, i64* %207, i64 2
  %211 = bitcast i64* %210 to <2 x i64>*
  %212 = load <2 x i64>, <2 x i64>* %211, align 8, !tbaa !29, !alias.scope !46, !noalias !43
  %213 = bitcast %"class.std::thread"* %206 to <2 x i64>*
  store <2 x i64> %209, <2 x i64>* %213, align 8, !tbaa !29, !alias.scope !43, !noalias !46
  %214 = getelementptr %"class.std::thread", %"class.std::thread"* %206, i64 2
  %215 = bitcast %"class.std::thread"* %214 to <2 x i64>*
  store <2 x i64> %212, <2 x i64>* %215, align 8, !tbaa !29, !alias.scope !43, !noalias !46
  %216 = bitcast i64* %207 to <2 x i64>*
  store <2 x i64> zeroinitializer, <2 x i64>* %216, align 8, !tbaa !29, !alias.scope !46, !noalias !43
  %217 = bitcast i64* %210 to <2 x i64>*
  store <2 x i64> zeroinitializer, <2 x i64>* %217, align 8, !tbaa !29, !alias.scope !46, !noalias !43
  br label %218

218:                                              ; preds = %202, %205
  %219 = icmp eq i64 %158, %161
  br i1 %219, label %232, label %220

220:                                              ; preds = %154, %218
  %221 = phi %"class.std::thread"* [ %152, %154 ], [ %162, %218 ]
  %222 = phi %"class.std::thread"* [ %1, %154 ], [ %163, %218 ]
  br label %223

223:                                              ; preds = %220, %223
  %224 = phi %"class.std::thread"* [ %230, %223 ], [ %221, %220 ]
  %225 = phi %"class.std::thread"* [ %229, %223 ], [ %222, %220 ]
  call void @llvm.experimental.noalias.scope.decl(metadata !43) #14
  call void @llvm.experimental.noalias.scope.decl(metadata !46) #14
  %226 = getelementptr inbounds %"class.std::thread", %"class.std::thread"* %224, i64 0, i32 0, i32 0
  %227 = getelementptr inbounds %"class.std::thread", %"class.std::thread"* %225, i64 0, i32 0, i32 0
  %228 = load i64, i64* %227, align 8, !tbaa !29, !alias.scope !46, !noalias !43
  store i64 %228, i64* %226, align 8, !tbaa !29, !alias.scope !43, !noalias !46
  store i64 0, i64* %227, align 8, !tbaa !29, !alias.scope !46, !noalias !43
  %229 = getelementptr inbounds %"class.std::thread", %"class.std::thread"* %225, i64 1
  %230 = getelementptr inbounds %"class.std::thread", %"class.std::thread"* %224, i64 1
  %231 = icmp eq %"class.std::thread"* %229, %9
  br i1 %231, label %232, label %223, !llvm.loop !53

232:                                              ; preds = %223, %218, %150
  %233 = phi %"class.std::thread"* [ %152, %150 ], [ %162, %218 ], [ %230, %223 ]
  %234 = icmp eq %"class.std::thread"* %12, null
  br i1 %234, label %237, label %235

235:                                              ; preds = %232
  %236 = bitcast %"class.std::thread"* %12 to i8*
  call void @_ZdlPv(i8* noundef %236) #17
  br label %237

237:                                              ; preds = %232, %235
  %238 = getelementptr inbounds %"class.std::vector", %"class.std::vector"* %0, i64 0, i32 0, i32 0, i32 0, i32 2
  store %"class.std::thread"* %37, %"class.std::thread"** %11, align 8, !tbaa !25
  store %"class.std::thread"* %233, %"class.std::thread"** %8, align 8, !tbaa !11
  %239 = getelementptr inbounds %"class.std::thread", %"class.std::thread"* %37, i64 %27
  store %"class.std::thread"* %239, %"class.std::thread"** %238, align 8, !tbaa !13
  ret void

240:                                              ; preds = %36
  %241 = landingpad { i8*, i32 }
          catch i8* null
  br label %244

242:                                              ; preds = %244
  %243 = landingpad { i8*, i32 }
          cleanup
  invoke void @__cxa_end_catch()
          to label %249 unwind label %250

244:                                              ; preds = %240, %65, %61
  %245 = phi { i8*, i32 } [ %241, %240 ], [ %62, %65 ], [ %62, %61 ]
  %246 = extractvalue { i8*, i32 } %245, 0
  %247 = call i8* @__cxa_begin_catch(i8* %246) #14
  %248 = bitcast %"class.std::thread"* %37 to i8*
  call void @_ZdlPv(i8* noundef %248) #17
  invoke void @__cxa_rethrow() #18
          to label %253 unwind label %242

249:                                              ; preds = %242
  resume { i8*, i32 } %243

250:                                              ; preds = %242
  %251 = landingpad { i8*, i32 }
          catch i8* null
  %252 = extractvalue { i8*, i32 } %251, 0
  call void @__clang_call_terminate(i8* %252) #16
  unreachable

253:                                              ; preds = %244
  unreachable
}

declare void @_ZNSt6thread15_M_start_threadESt10unique_ptrINS_6_StateESt14default_deleteIS1_EEPFvvE(%"class.std::thread"* noundef nonnull align 8 dereferenceable(8), %"class.std::unique_ptr"* noundef, void ()* noundef) local_unnamed_addr #0

; Function Attrs: nobuiltin allocsize(0)
declare noundef nonnull i8* @_Znwm(i64 noundef) local_unnamed_addr #8

; Function Attrs: nounwind
declare void @_ZNSt6thread6_StateD2Ev(%"struct.std::thread::_State"* noundef nonnull align 8 dereferenceable(8)) unnamed_addr #1

; Function Attrs: inlinehint nounwind uwtable
define linkonce_odr dso_local void @_ZNSt6thread11_State_implINS_8_InvokerISt5tupleIJPFviiEiiEEEEED0Ev(%"struct.std::thread::_State_impl"* noundef nonnull align 8 dereferenceable(24) %0) unnamed_addr #9 comdat align 2 {
  %2 = getelementptr inbounds %"struct.std::thread::_State_impl", %"struct.std::thread::_State_impl"* %0, i64 0, i32 0
  tail call void @_ZNSt6thread6_StateD2Ev(%"struct.std::thread::_State"* noundef nonnull align 8 dereferenceable(24) %2) #14
  %3 = bitcast %"struct.std::thread::_State_impl"* %0 to i8*
  tail call void @_ZdlPv(i8* noundef nonnull %3) #17
  ret void
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @_ZNSt6thread11_State_implINS_8_InvokerISt5tupleIJPFviiEiiEEEEE6_M_runEv(%"struct.std::thread::_State_impl"* noundef nonnull align 8 dereferenceable(24) %0) unnamed_addr #10 comdat align 2 {
  %2 = getelementptr inbounds %"struct.std::thread::_State_impl", %"struct.std::thread::_State_impl"* %0, i64 0, i32 1, i32 0, i32 0, i32 1, i32 0
  %3 = getelementptr inbounds %"struct.std::thread::_State_impl", %"struct.std::thread::_State_impl"* %0, i64 0, i32 1, i32 0, i32 0, i32 0, i32 1, i32 0
  %4 = getelementptr inbounds %"struct.std::thread::_State_impl", %"struct.std::thread::_State_impl"* %0, i64 0, i32 1, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0
  %5 = load void (i32, i32)*, void (i32, i32)** %2, align 8, !tbaa !5
  %6 = load i32, i32* %3, align 4, !tbaa !9
  %7 = load i32, i32* %4, align 8, !tbaa !9
  tail call void %5(i32 noundef %6, i32 noundef %7)
  ret void
}

declare void @__cxa_rethrow() local_unnamed_addr

declare void @__cxa_end_catch() local_unnamed_addr

; Function Attrs: noreturn
declare void @_ZSt20__throw_length_errorPKc(i8* noundef) local_unnamed_addr #11

; Function Attrs: uwtable
define internal void @_GLOBAL__sub_I_parallelFor.cpp() #3 section ".text.startup" {
  tail call void @_ZNSt8ios_base4InitC1Ev(%"class.std::ios_base::Init"* noundef nonnull align 1 dereferenceable(1) @_ZStL8__ioinit)
  %1 = tail call i32 @__cxa_atexit(void (i8*)* bitcast (void (%"class.std::ios_base::Init"*)* @_ZNSt8ios_base4InitD1Ev to void (i8*)*), i8* getelementptr inbounds (%"class.std::ios_base::Init", %"class.std::ios_base::Init"* @_ZStL8__ioinit, i64 0, i32 0), i8* nonnull @__dso_handle) #14
  ret void
}

; Function Attrs: argmemonly nofree nounwind willreturn writeonly
declare void @llvm.memset.p0i8.i64(i8* nocapture writeonly, i8, i64, i1 immarg) #12

; Function Attrs: inaccessiblememonly nofree nosync nounwind willreturn
declare void @llvm.experimental.noalias.scope.decl(metadata) #13

attributes #0 = { "approx-func-fp-math"="true" "denormal-fp-math"="preserve-sign,preserve-sign" "denormal-fp-math-f32"="ieee,ieee" "frame-pointer"="none" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "unsafe-fp-math"="true" }
attributes #1 = { nounwind "approx-func-fp-math"="true" "denormal-fp-math"="preserve-sign,preserve-sign" "denormal-fp-math-f32"="ieee,ieee" "frame-pointer"="none" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "unsafe-fp-math"="true" }
attributes #2 = { nofree nounwind }
attributes #3 = { uwtable "approx-func-fp-math"="true" "denormal-fp-math"="preserve-sign,preserve-sign" "denormal-fp-math-f32"="ieee,ieee" "frame-pointer"="none" "min-legal-vector-width"="0" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "unsafe-fp-math"="true" }
attributes #4 = { argmemonly mustprogress nofree nosync nounwind willreturn }
attributes #5 = { nounwind uwtable "approx-func-fp-math"="true" "denormal-fp-math"="preserve-sign,preserve-sign" "denormal-fp-math-f32"="ieee,ieee" "frame-pointer"="none" "min-legal-vector-width"="0" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "unsafe-fp-math"="true" }
attributes #6 = { noinline noreturn nounwind }
attributes #7 = { nobuiltin nounwind "approx-func-fp-math"="true" "denormal-fp-math"="preserve-sign,preserve-sign" "denormal-fp-math-f32"="ieee,ieee" "frame-pointer"="none" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "unsafe-fp-math"="true" }
attributes #8 = { nobuiltin allocsize(0) "approx-func-fp-math"="true" "denormal-fp-math"="preserve-sign,preserve-sign" "denormal-fp-math-f32"="ieee,ieee" "frame-pointer"="none" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "unsafe-fp-math"="true" }
attributes #9 = { inlinehint nounwind uwtable "approx-func-fp-math"="true" "denormal-fp-math"="preserve-sign,preserve-sign" "denormal-fp-math-f32"="ieee,ieee" "frame-pointer"="none" "min-legal-vector-width"="0" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "unsafe-fp-math"="true" }
attributes #10 = { mustprogress uwtable "approx-func-fp-math"="true" "denormal-fp-math"="preserve-sign,preserve-sign" "denormal-fp-math-f32"="ieee,ieee" "frame-pointer"="none" "min-legal-vector-width"="0" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "unsafe-fp-math"="true" }
attributes #11 = { noreturn "approx-func-fp-math"="true" "denormal-fp-math"="preserve-sign,preserve-sign" "denormal-fp-math-f32"="ieee,ieee" "frame-pointer"="none" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "unsafe-fp-math"="true" }
attributes #12 = { argmemonly nofree nounwind willreturn writeonly }
attributes #13 = { inaccessiblememonly nofree nosync nounwind willreturn }
attributes #14 = { nounwind }
attributes #15 = { builtin allocsize(0) }
attributes #16 = { noreturn nounwind }
attributes #17 = { builtin nounwind }
attributes #18 = { noreturn }

!llvm.module.flags = !{!0, !1, !2, !3}
!llvm.ident = !{!4}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 1}
!4 = !{!"Debian clang version 14.0.6"}
!5 = !{!6, !6, i64 0}
!6 = !{!"any pointer", !7, i64 0}
!7 = !{!"omnipotent char", !8, i64 0}
!8 = !{!"Simple C++ TBAA"}
!9 = !{!10, !10, i64 0}
!10 = !{!"int", !7, i64 0}
!11 = !{!12, !6, i64 8}
!12 = !{!"_ZTSNSt12_Vector_baseISt6threadSaIS0_EE17_Vector_impl_dataE", !6, i64 0, !6, i64 8, !6, i64 16}
!13 = !{!12, !6, i64 16}
!14 = !{!15, !16, i64 0}
!15 = !{!"_ZTSNSt6thread2idE", !16, i64 0}
!16 = !{!"long", !7, i64 0}
!17 = !{!18, !18, i64 0}
!18 = !{!"vtable pointer", !8, i64 0}
!19 = !{!20, !10, i64 0}
!20 = !{!"_ZTSSt10_Head_baseILm2EiLb0EE", !10, i64 0}
!21 = !{!22, !10, i64 0}
!22 = !{!"_ZTSSt10_Head_baseILm1EiLb0EE", !10, i64 0}
!23 = !{!24, !6, i64 0}
!24 = !{!"_ZTSSt10_Head_baseILm0EPFviiELb0EE", !6, i64 0}
!25 = !{!12, !6, i64 0}
!26 = distinct !{!26, !27}
!27 = !{!"llvm.loop.mustprogress"}
!28 = !{i64 0, i64 8, !29}
!29 = !{!16, !16, i64 0}
!30 = !{!31}
!31 = distinct !{!31, !32, !"_ZSt19__relocate_object_aISt6threadS0_SaIS0_EEvPT_PT0_RT1_: argument 0"}
!32 = distinct !{!32, !"_ZSt19__relocate_object_aISt6threadS0_SaIS0_EEvPT_PT0_RT1_"}
!33 = !{!34}
!34 = distinct !{!34, !32, !"_ZSt19__relocate_object_aISt6threadS0_SaIS0_EEvPT_PT0_RT1_: argument 1"}
!35 = !{!36}
!36 = distinct !{!36, !32, !"_ZSt19__relocate_object_aISt6threadS0_SaIS0_EEvPT_PT0_RT1_: argument 0:It1"}
!37 = !{!38}
!38 = distinct !{!38, !32, !"_ZSt19__relocate_object_aISt6threadS0_SaIS0_EEvPT_PT0_RT1_: argument 1:It1"}
!39 = distinct !{!39, !27, !40}
!40 = !{!"llvm.loop.isvectorized", i32 1}
!41 = distinct !{!41, !27, !42, !40}
!42 = !{!"llvm.loop.unroll.runtime.disable"}
!43 = !{!44}
!44 = distinct !{!44, !45, !"_ZSt19__relocate_object_aISt6threadS0_SaIS0_EEvPT_PT0_RT1_: argument 0"}
!45 = distinct !{!45, !"_ZSt19__relocate_object_aISt6threadS0_SaIS0_EEvPT_PT0_RT1_"}
!46 = !{!47}
!47 = distinct !{!47, !45, !"_ZSt19__relocate_object_aISt6threadS0_SaIS0_EEvPT_PT0_RT1_: argument 1"}
!48 = !{!49}
!49 = distinct !{!49, !45, !"_ZSt19__relocate_object_aISt6threadS0_SaIS0_EEvPT_PT0_RT1_: argument 0:It1"}
!50 = !{!51}
!51 = distinct !{!51, !45, !"_ZSt19__relocate_object_aISt6threadS0_SaIS0_EEvPT_PT0_RT1_: argument 1:It1"}
!52 = distinct !{!52, !27, !40}
!53 = distinct !{!53, !27, !42, !40}
