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
  %9 = icmp slt i32 %0, %1
  %10 = sub nsw i32 %1, %0
  %11 = sub nsw i32 %0, %1
  %12 = select i1 %9, i32 %10, i32 %11
  %13 = sdiv i32 %12, 4
  %14 = srem i32 %12, 4
  %15 = bitcast %"class.std::vector"* %6 to i8*
  call void @llvm.lifetime.start.p0i8(i64 24, i8* nonnull %15) #14
  call void @llvm.memset.p0i8.i64(i8* noundef nonnull align 8 dereferenceable(24) %15, i8 0, i64 24, i1 false) #14
  %16 = bitcast i32* %7 to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* nonnull %16) #14
  store i32 %0, i32* %7, align 4, !tbaa !9
  %17 = bitcast i32* %8 to i8*
  %18 = getelementptr inbounds %"class.std::vector", %"class.std::vector"* %6, i64 0, i32 0, i32 0, i32 0, i32 1
  %19 = getelementptr inbounds %"class.std::vector", %"class.std::vector"* %6, i64 0, i32 0, i32 0, i32 0, i32 2
  %20 = bitcast %"class.std::unique_ptr"* %4 to i8*
  %21 = getelementptr inbounds %"class.std::unique_ptr", %"class.std::unique_ptr"* %4, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0
  br label %28

22:                                               ; preds = %95
  %23 = getelementptr inbounds %"class.std::vector", %"class.std::vector"* %6, i64 0, i32 0, i32 0, i32 0, i32 0
  %24 = load %"class.std::thread"*, %"class.std::thread"** %23, align 8, !tbaa !5
  %25 = load %"class.std::thread"*, %"class.std::thread"** %18, align 8, !tbaa !5
  %26 = icmp eq %"class.std::thread"* %24, %25
  br i1 %26, label %27, label %122

27:                                               ; preds = %22
  call void @llvm.lifetime.end.p0i8(i64 4, i8* nonnull %16) #14
  br label %116

28:                                               ; preds = %3, %95
  %29 = phi i32 [ %0, %3 ], [ %96, %95 ]
  %30 = phi i32 [ 0, %3 ], [ %97, %95 ]
  call void @llvm.lifetime.start.p0i8(i64 4, i8* nonnull %17) #14
  br i1 %9, label %31, label %36

31:                                               ; preds = %28
  %32 = add nsw i32 %29, %13
  store i32 %32, i32* %8, align 4, !tbaa !9
  %33 = icmp slt i32 %30, %14
  br i1 %33, label %34, label %41

34:                                               ; preds = %31
  %35 = add nsw i32 %32, 1
  store i32 %35, i32* %8, align 4, !tbaa !9
  br label %41

36:                                               ; preds = %28
  %37 = sub nsw i32 %29, %13
  store i32 %37, i32* %8, align 4, !tbaa !9
  %38 = icmp slt i32 %30, %14
  br i1 %38, label %39, label %41

39:                                               ; preds = %36
  %40 = add nsw i32 %37, -1
  store i32 %40, i32* %8, align 4, !tbaa !9
  br label %41

41:                                               ; preds = %36, %39, %31, %34
  %42 = invoke noundef nonnull align 8 dereferenceable(8) %"class.std::basic_ostream"* @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(%"class.std::basic_ostream"* noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, i8* noundef nonnull getelementptr inbounds ([18 x i8], [18 x i8]* @.str, i64 0, i64 0), i64 noundef 17)
          to label %43 unwind label %99

43:                                               ; preds = %41
  %44 = invoke noundef nonnull align 8 dereferenceable(8) %"class.std::basic_ostream"* @_ZNSolsEi(%"class.std::basic_ostream"* noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, i32 noundef %30)
          to label %45 unwind label %99

45:                                               ; preds = %43
  %46 = invoke noundef nonnull align 8 dereferenceable(8) %"class.std::basic_ostream"* @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(%"class.std::basic_ostream"* noundef nonnull align 8 dereferenceable(8) %44, i8* noundef nonnull getelementptr inbounds ([14 x i8], [14 x i8]* @.str.1, i64 0, i64 0), i64 noundef 13)
          to label %47 unwind label %99

47:                                               ; preds = %45
  %48 = load i32, i32* %7, align 4, !tbaa !9
  %49 = invoke noundef nonnull align 8 dereferenceable(8) %"class.std::basic_ostream"* @_ZNSolsEi(%"class.std::basic_ostream"* noundef nonnull align 8 dereferenceable(8) %44, i32 noundef %48)
          to label %50 unwind label %99

50:                                               ; preds = %47
  %51 = invoke noundef nonnull align 8 dereferenceable(8) %"class.std::basic_ostream"* @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(%"class.std::basic_ostream"* noundef nonnull align 8 dereferenceable(8) %49, i8* noundef nonnull getelementptr inbounds ([3 x i8], [3 x i8]* @.str.2, i64 0, i64 0), i64 noundef 2)
          to label %52 unwind label %99

52:                                               ; preds = %50
  %53 = load i32, i32* %8, align 4, !tbaa !9
  %54 = invoke noundef nonnull align 8 dereferenceable(8) %"class.std::basic_ostream"* @_ZNSolsEi(%"class.std::basic_ostream"* noundef nonnull align 8 dereferenceable(8) %49, i32 noundef %53)
          to label %55 unwind label %99

55:                                               ; preds = %52
  %56 = invoke noundef nonnull align 8 dereferenceable(8) %"class.std::basic_ostream"* @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(%"class.std::basic_ostream"* noundef nonnull align 8 dereferenceable(8) %54, i8* noundef nonnull getelementptr inbounds ([3 x i8], [3 x i8]* @.str.3, i64 0, i64 0), i64 noundef 2)
          to label %57 unwind label %99

57:                                               ; preds = %55
  %58 = load %"class.std::thread"*, %"class.std::thread"** %18, align 8, !tbaa !11
  %59 = load %"class.std::thread"*, %"class.std::thread"** %19, align 8, !tbaa !13
  %60 = icmp eq %"class.std::thread"* %58, %59
  br i1 %60, label %94, label %61

61:                                               ; preds = %57
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %20)
  %62 = getelementptr inbounds %"class.std::thread", %"class.std::thread"* %58, i64 0, i32 0, i32 0
  store i64 0, i64* %62, align 8, !tbaa !14
  %63 = invoke noalias noundef nonnull dereferenceable(24) i8* @_Znwm(i64 noundef 24) #15
          to label %64 unwind label %99

64:                                               ; preds = %61
  %65 = bitcast i8* %63 to %"struct.std::thread::_State_impl"*
  %66 = getelementptr inbounds %"struct.std::thread::_State_impl", %"struct.std::thread::_State_impl"* %65, i64 0, i32 0, i32 0
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [5 x i8*] }, { [5 x i8*] }* @_ZTVNSt6thread11_State_implINS_8_InvokerISt5tupleIJPFviiEiiEEEEEE, i64 0, inrange i32 0, i64 2) to i32 (...)**), i32 (...)*** %66, align 8, !tbaa !17
  %67 = getelementptr inbounds %"struct.std::thread::_State_impl", %"struct.std::thread::_State_impl"* %65, i64 0, i32 1, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0
  %68 = load i32, i32* %8, align 4, !tbaa !9
  store i32 %68, i32* %67, align 8, !tbaa !19
  %69 = getelementptr inbounds %"struct.std::thread::_State_impl", %"struct.std::thread::_State_impl"* %65, i64 0, i32 1, i32 0, i32 0, i32 0, i32 1, i32 0
  %70 = load i32, i32* %7, align 4, !tbaa !9
  store i32 %70, i32* %69, align 4, !tbaa !21
  %71 = getelementptr inbounds %"struct.std::thread::_State_impl", %"struct.std::thread::_State_impl"* %65, i64 0, i32 1, i32 0, i32 0, i32 1, i32 0
  %72 = load void (i32, i32)*, void (i32, i32)** %5, align 8, !tbaa !5
  store void (i32, i32)* %72, void (i32, i32)** %71, align 8, !tbaa !23
  %73 = getelementptr %"struct.std::thread::_State_impl", %"struct.std::thread::_State_impl"* %65, i64 0, i32 0
  store %"struct.std::thread::_State"* %73, %"struct.std::thread::_State"** %21, align 8, !tbaa !5
  invoke void @_ZNSt6thread15_M_start_threadESt10unique_ptrINS_6_StateESt14default_deleteIS1_EEPFvvE(%"class.std::thread"* noundef nonnull align 8 dereferenceable(8) %58, %"class.std::unique_ptr"* noundef nonnull %4, void ()* noundef null)
          to label %74 unwind label %82

74:                                               ; preds = %64
  %75 = load %"struct.std::thread::_State"*, %"struct.std::thread::_State"** %21, align 8, !tbaa !5
  %76 = icmp eq %"struct.std::thread::_State"* %75, null
  br i1 %76, label %91, label %77

77:                                               ; preds = %74
  %78 = bitcast %"struct.std::thread::_State"* %75 to void (%"struct.std::thread::_State"*)***
  %79 = load void (%"struct.std::thread::_State"*)**, void (%"struct.std::thread::_State"*)*** %78, align 8, !tbaa !17
  %80 = getelementptr inbounds void (%"struct.std::thread::_State"*)*, void (%"struct.std::thread::_State"*)** %79, i64 1
  %81 = load void (%"struct.std::thread::_State"*)*, void (%"struct.std::thread::_State"*)** %80, align 8
  call void %81(%"struct.std::thread::_State"* noundef nonnull align 8 dereferenceable(8) %75) #14
  br label %91

82:                                               ; preds = %64
  %83 = landingpad { i8*, i32 }
          cleanup
  %84 = load %"struct.std::thread::_State"*, %"struct.std::thread::_State"** %21, align 8, !tbaa !5
  %85 = icmp eq %"struct.std::thread::_State"* %84, null
  br i1 %85, label %101, label %86

86:                                               ; preds = %82
  %87 = bitcast %"struct.std::thread::_State"* %84 to void (%"struct.std::thread::_State"*)***
  %88 = load void (%"struct.std::thread::_State"*)**, void (%"struct.std::thread::_State"*)*** %87, align 8, !tbaa !17
  %89 = getelementptr inbounds void (%"struct.std::thread::_State"*)*, void (%"struct.std::thread::_State"*)** %88, i64 1
  %90 = load void (%"struct.std::thread::_State"*)*, void (%"struct.std::thread::_State"*)** %89, align 8
  call void %90(%"struct.std::thread::_State"* noundef nonnull align 8 dereferenceable(8) %84) #14
  br label %101

91:                                               ; preds = %77, %74
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %20)
  %92 = load %"class.std::thread"*, %"class.std::thread"** %18, align 8, !tbaa !11
  %93 = getelementptr inbounds %"class.std::thread", %"class.std::thread"* %92, i64 1
  store %"class.std::thread"* %93, %"class.std::thread"** %18, align 8, !tbaa !11
  br label %95

94:                                               ; preds = %57
  invoke void @_ZNSt6vectorISt6threadSaIS0_EE17_M_realloc_insertIJRPFviiERiS7_EEEvN9__gnu_cxx17__normal_iteratorIPS0_S2_EEDpOT_(%"class.std::vector"* noundef nonnull align 8 dereferenceable(24) %6, %"class.std::thread"* %58, void (i32, i32)** noundef nonnull align 8 dereferenceable(8) %5, i32* noundef nonnull align 4 dereferenceable(4) %7, i32* noundef nonnull align 4 dereferenceable(4) %8)
          to label %95 unwind label %99

95:                                               ; preds = %91, %94
  %96 = load i32, i32* %8, align 4, !tbaa !9
  store i32 %96, i32* %7, align 4, !tbaa !9
  call void @llvm.lifetime.end.p0i8(i64 4, i8* nonnull %17) #14
  %97 = add nuw nsw i32 %30, 1
  %98 = icmp eq i32 %97, 4
  br i1 %98, label %22, label %28, !llvm.loop !25

99:                                               ; preds = %94, %61, %55, %50, %45, %41, %52, %47, %43
  %100 = landingpad { i8*, i32 }
          cleanup
  br label %101

101:                                              ; preds = %82, %86, %99
  %102 = phi { i8*, i32 } [ %100, %99 ], [ %83, %86 ], [ %83, %82 ]
  call void @llvm.lifetime.end.p0i8(i64 4, i8* nonnull %17) #14
  br label %129

103:                                              ; preds = %124
  %104 = load %"class.std::thread"*, %"class.std::thread"** %23, align 8, !tbaa !27
  %105 = load %"class.std::thread"*, %"class.std::thread"** %18, align 8, !tbaa !11
  call void @llvm.lifetime.end.p0i8(i64 4, i8* nonnull %16) #14
  %106 = icmp eq %"class.std::thread"* %104, %105
  br i1 %106, label %116, label %109

107:                                              ; preds = %109
  %108 = icmp eq %"class.std::thread"* %114, %105
  br i1 %108, label %116, label %109, !llvm.loop !28

109:                                              ; preds = %103, %107
  %110 = phi %"class.std::thread"* [ %114, %107 ], [ %104, %103 ]
  %111 = getelementptr inbounds %"class.std::thread", %"class.std::thread"* %110, i64 0, i32 0, i32 0
  %112 = load i64, i64* %111, align 8, !tbaa.struct !29
  %113 = icmp eq i64 %112, 0
  %114 = getelementptr inbounds %"class.std::thread", %"class.std::thread"* %110, i64 1
  br i1 %113, label %107, label %115

115:                                              ; preds = %109
  call void @_ZSt9terminatev() #16
  unreachable

116:                                              ; preds = %107, %27, %103
  %117 = phi %"class.std::thread"* [ %24, %27 ], [ %104, %103 ], [ %104, %107 ]
  %118 = icmp eq %"class.std::thread"* %117, null
  br i1 %118, label %121, label %119

119:                                              ; preds = %116
  %120 = bitcast %"class.std::thread"* %117 to i8*
  call void @_ZdlPv(i8* noundef %120) #17
  br label %121

121:                                              ; preds = %116, %119
  call void @llvm.lifetime.end.p0i8(i64 24, i8* nonnull %15) #14
  ret void

122:                                              ; preds = %22, %124
  %123 = phi %"class.std::thread"* [ %125, %124 ], [ %24, %22 ]
  invoke void @_ZNSt6thread4joinEv(%"class.std::thread"* noundef nonnull align 8 dereferenceable(8) %123)
          to label %124 unwind label %127

124:                                              ; preds = %122
  %125 = getelementptr inbounds %"class.std::thread", %"class.std::thread"* %123, i64 1
  %126 = icmp eq %"class.std::thread"* %125, %25
  br i1 %126, label %103, label %122

127:                                              ; preds = %122
  %128 = landingpad { i8*, i32 }
          cleanup
  br label %129

129:                                              ; preds = %127, %101
  %130 = phi { i8*, i32 } [ %102, %101 ], [ %128, %127 ]
  call void @llvm.lifetime.end.p0i8(i64 4, i8* nonnull %16) #14
  call void @_ZNSt6vectorISt6threadSaIS0_EED2Ev(%"class.std::vector"* noundef nonnull align 8 dereferenceable(24) %6) #14
  call void @llvm.lifetime.end.p0i8(i64 24, i8* nonnull %15) #14
  resume { i8*, i32 } %130
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
  %3 = load %"class.std::thread"*, %"class.std::thread"** %2, align 8, !tbaa !27
  %4 = getelementptr inbounds %"class.std::vector", %"class.std::vector"* %0, i64 0, i32 0, i32 0, i32 0, i32 1
  %5 = load %"class.std::thread"*, %"class.std::thread"** %4, align 8, !tbaa !11
  %6 = icmp eq %"class.std::thread"* %3, %5
  br i1 %6, label %16, label %9

7:                                                ; preds = %9
  %8 = icmp eq %"class.std::thread"* %14, %5
  br i1 %8, label %16, label %9, !llvm.loop !28

9:                                                ; preds = %1, %7
  %10 = phi %"class.std::thread"* [ %14, %7 ], [ %3, %1 ]
  %11 = getelementptr inbounds %"class.std::thread", %"class.std::thread"* %10, i64 0, i32 0, i32 0
  %12 = load i64, i64* %11, align 8, !tbaa.struct !29
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
  %12 = load %"class.std::thread"*, %"class.std::thread"** %11, align 8, !tbaa !27
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
  call void @llvm.experimental.noalias.scope.decl(metadata !31) #14
  call void @llvm.experimental.noalias.scope.decl(metadata !34) #14
  %92 = getelementptr %"class.std::thread", %"class.std::thread"* %37, i64 %90, i32 0, i32 0
  %93 = getelementptr %"class.std::thread", %"class.std::thread"* %12, i64 %90, i32 0, i32 0
  %94 = bitcast i64* %93 to <2 x i64>*
  %95 = load <2 x i64>, <2 x i64>* %94, align 8, !tbaa !30, !alias.scope !34, !noalias !31
  %96 = getelementptr i64, i64* %93, i64 2
  %97 = bitcast i64* %96 to <2 x i64>*
  %98 = load <2 x i64>, <2 x i64>* %97, align 8, !tbaa !30, !alias.scope !34, !noalias !31
  %99 = bitcast i64* %92 to <2 x i64>*
  store <2 x i64> %95, <2 x i64>* %99, align 8, !tbaa !30, !alias.scope !31, !noalias !34
  %100 = getelementptr i64, i64* %92, i64 2
  %101 = bitcast i64* %100 to <2 x i64>*
  store <2 x i64> %98, <2 x i64>* %101, align 8, !tbaa !30, !alias.scope !31, !noalias !34
  %102 = bitcast i64* %93 to <2 x i64>*
  store <2 x i64> zeroinitializer, <2 x i64>* %102, align 8, !tbaa !30, !alias.scope !34, !noalias !31
  %103 = bitcast i64* %96 to <2 x i64>*
  store <2 x i64> zeroinitializer, <2 x i64>* %103, align 8, !tbaa !30, !alias.scope !34, !noalias !31
  %104 = or i64 %90, 4
  call void @llvm.experimental.noalias.scope.decl(metadata !36) #14
  call void @llvm.experimental.noalias.scope.decl(metadata !38) #14
  %105 = getelementptr %"class.std::thread", %"class.std::thread"* %37, i64 %104, i32 0, i32 0
  %106 = getelementptr %"class.std::thread", %"class.std::thread"* %12, i64 %104, i32 0, i32 0
  %107 = bitcast i64* %106 to <2 x i64>*
  %108 = load <2 x i64>, <2 x i64>* %107, align 8, !tbaa !30, !alias.scope !38, !noalias !36
  %109 = getelementptr i64, i64* %106, i64 2
  %110 = bitcast i64* %109 to <2 x i64>*
  %111 = load <2 x i64>, <2 x i64>* %110, align 8, !tbaa !30, !alias.scope !38, !noalias !36
  %112 = bitcast i64* %105 to <2 x i64>*
  store <2 x i64> %108, <2 x i64>* %112, align 8, !tbaa !30, !alias.scope !36, !noalias !38
  %113 = getelementptr i64, i64* %105, i64 2
  %114 = bitcast i64* %113 to <2 x i64>*
  store <2 x i64> %111, <2 x i64>* %114, align 8, !tbaa !30, !alias.scope !36, !noalias !38
  %115 = bitcast i64* %106 to <2 x i64>*
  store <2 x i64> zeroinitializer, <2 x i64>* %115, align 8, !tbaa !30, !alias.scope !38, !noalias !36
  %116 = bitcast i64* %109 to <2 x i64>*
  store <2 x i64> zeroinitializer, <2 x i64>* %116, align 8, !tbaa !30, !alias.scope !38, !noalias !36
  %117 = add nuw i64 %90, 8
  %118 = add i64 %91, 2
  %119 = icmp eq i64 %118, %88
  br i1 %119, label %120, label %89, !llvm.loop !40

120:                                              ; preds = %89, %78
  %121 = phi i64 [ 0, %78 ], [ %117, %89 ]
  %122 = icmp eq i64 %85, 0
  br i1 %122, label %136, label %123

123:                                              ; preds = %120
  call void @llvm.experimental.noalias.scope.decl(metadata !31) #14
  call void @llvm.experimental.noalias.scope.decl(metadata !34) #14
  %124 = getelementptr %"class.std::thread", %"class.std::thread"* %37, i64 %121, i32 0, i32 0
  %125 = getelementptr %"class.std::thread", %"class.std::thread"* %12, i64 %121, i32 0, i32 0
  %126 = bitcast i64* %125 to <2 x i64>*
  %127 = load <2 x i64>, <2 x i64>* %126, align 8, !tbaa !30, !alias.scope !34, !noalias !31
  %128 = getelementptr i64, i64* %125, i64 2
  %129 = bitcast i64* %128 to <2 x i64>*
  %130 = load <2 x i64>, <2 x i64>* %129, align 8, !tbaa !30, !alias.scope !34, !noalias !31
  %131 = bitcast i64* %124 to <2 x i64>*
  store <2 x i64> %127, <2 x i64>* %131, align 8, !tbaa !30, !alias.scope !31, !noalias !34
  %132 = getelementptr i64, i64* %124, i64 2
  %133 = bitcast i64* %132 to <2 x i64>*
  store <2 x i64> %130, <2 x i64>* %133, align 8, !tbaa !30, !alias.scope !31, !noalias !34
  %134 = bitcast i64* %125 to <2 x i64>*
  store <2 x i64> zeroinitializer, <2 x i64>* %134, align 8, !tbaa !30, !alias.scope !34, !noalias !31
  %135 = bitcast i64* %128 to <2 x i64>*
  store <2 x i64> zeroinitializer, <2 x i64>* %135, align 8, !tbaa !30, !alias.scope !34, !noalias !31
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
  call void @llvm.experimental.noalias.scope.decl(metadata !31) #14
  call void @llvm.experimental.noalias.scope.decl(metadata !34) #14
  %144 = getelementptr inbounds %"class.std::thread", %"class.std::thread"* %142, i64 0, i32 0, i32 0
  %145 = getelementptr inbounds %"class.std::thread", %"class.std::thread"* %143, i64 0, i32 0, i32 0
  %146 = load i64, i64* %145, align 8, !tbaa !30, !alias.scope !34, !noalias !31
  store i64 %146, i64* %144, align 8, !tbaa !30, !alias.scope !31, !noalias !34
  store i64 0, i64* %145, align 8, !tbaa !30, !alias.scope !34, !noalias !31
  %147 = getelementptr inbounds %"class.std::thread", %"class.std::thread"* %143, i64 1
  %148 = getelementptr inbounds %"class.std::thread", %"class.std::thread"* %142, i64 1
  %149 = icmp eq %"class.std::thread"* %147, %1
  br i1 %149, label %150, label %141, !llvm.loop !42

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
  call void @llvm.experimental.noalias.scope.decl(metadata !44) #14
  call void @llvm.experimental.noalias.scope.decl(metadata !47) #14
  %175 = getelementptr %"class.std::thread", %"class.std::thread"* %1, i64 %172, i32 0, i32 0
  %176 = bitcast i64* %175 to <2 x i64>*
  %177 = load <2 x i64>, <2 x i64>* %176, align 8, !tbaa !30, !alias.scope !47, !noalias !44
  %178 = getelementptr i64, i64* %175, i64 2
  %179 = bitcast i64* %178 to <2 x i64>*
  %180 = load <2 x i64>, <2 x i64>* %179, align 8, !tbaa !30, !alias.scope !47, !noalias !44
  %181 = bitcast %"class.std::thread"* %174 to <2 x i64>*
  store <2 x i64> %177, <2 x i64>* %181, align 8, !tbaa !30, !alias.scope !44, !noalias !47
  %182 = getelementptr %"class.std::thread", %"class.std::thread"* %174, i64 2
  %183 = bitcast %"class.std::thread"* %182 to <2 x i64>*
  store <2 x i64> %180, <2 x i64>* %183, align 8, !tbaa !30, !alias.scope !44, !noalias !47
  %184 = bitcast i64* %175 to <2 x i64>*
  store <2 x i64> zeroinitializer, <2 x i64>* %184, align 8, !tbaa !30, !alias.scope !47, !noalias !44
  %185 = bitcast i64* %178 to <2 x i64>*
  store <2 x i64> zeroinitializer, <2 x i64>* %185, align 8, !tbaa !30, !alias.scope !47, !noalias !44
  %186 = or i64 %172, 4
  %187 = getelementptr %"class.std::thread", %"class.std::thread"* %152, i64 %186
  call void @llvm.experimental.noalias.scope.decl(metadata !49) #14
  call void @llvm.experimental.noalias.scope.decl(metadata !51) #14
  %188 = getelementptr %"class.std::thread", %"class.std::thread"* %1, i64 %186, i32 0, i32 0
  %189 = bitcast i64* %188 to <2 x i64>*
  %190 = load <2 x i64>, <2 x i64>* %189, align 8, !tbaa !30, !alias.scope !51, !noalias !49
  %191 = getelementptr i64, i64* %188, i64 2
  %192 = bitcast i64* %191 to <2 x i64>*
  %193 = load <2 x i64>, <2 x i64>* %192, align 8, !tbaa !30, !alias.scope !51, !noalias !49
  %194 = bitcast %"class.std::thread"* %187 to <2 x i64>*
  store <2 x i64> %190, <2 x i64>* %194, align 8, !tbaa !30, !alias.scope !49, !noalias !51
  %195 = getelementptr %"class.std::thread", %"class.std::thread"* %187, i64 2
  %196 = bitcast %"class.std::thread"* %195 to <2 x i64>*
  store <2 x i64> %193, <2 x i64>* %196, align 8, !tbaa !30, !alias.scope !49, !noalias !51
  %197 = bitcast i64* %188 to <2 x i64>*
  store <2 x i64> zeroinitializer, <2 x i64>* %197, align 8, !tbaa !30, !alias.scope !51, !noalias !49
  %198 = bitcast i64* %191 to <2 x i64>*
  store <2 x i64> zeroinitializer, <2 x i64>* %198, align 8, !tbaa !30, !alias.scope !51, !noalias !49
  %199 = add nuw i64 %172, 8
  %200 = add i64 %173, 2
  %201 = icmp eq i64 %200, %170
  br i1 %201, label %202, label %171, !llvm.loop !53

202:                                              ; preds = %171, %160
  %203 = phi i64 [ 0, %160 ], [ %199, %171 ]
  %204 = icmp eq i64 %167, 0
  br i1 %204, label %218, label %205

205:                                              ; preds = %202
  %206 = getelementptr %"class.std::thread", %"class.std::thread"* %152, i64 %203
  call void @llvm.experimental.noalias.scope.decl(metadata !44) #14
  call void @llvm.experimental.noalias.scope.decl(metadata !47) #14
  %207 = getelementptr %"class.std::thread", %"class.std::thread"* %1, i64 %203, i32 0, i32 0
  %208 = bitcast i64* %207 to <2 x i64>*
  %209 = load <2 x i64>, <2 x i64>* %208, align 8, !tbaa !30, !alias.scope !47, !noalias !44
  %210 = getelementptr i64, i64* %207, i64 2
  %211 = bitcast i64* %210 to <2 x i64>*
  %212 = load <2 x i64>, <2 x i64>* %211, align 8, !tbaa !30, !alias.scope !47, !noalias !44
  %213 = bitcast %"class.std::thread"* %206 to <2 x i64>*
  store <2 x i64> %209, <2 x i64>* %213, align 8, !tbaa !30, !alias.scope !44, !noalias !47
  %214 = getelementptr %"class.std::thread", %"class.std::thread"* %206, i64 2
  %215 = bitcast %"class.std::thread"* %214 to <2 x i64>*
  store <2 x i64> %212, <2 x i64>* %215, align 8, !tbaa !30, !alias.scope !44, !noalias !47
  %216 = bitcast i64* %207 to <2 x i64>*
  store <2 x i64> zeroinitializer, <2 x i64>* %216, align 8, !tbaa !30, !alias.scope !47, !noalias !44
  %217 = bitcast i64* %210 to <2 x i64>*
  store <2 x i64> zeroinitializer, <2 x i64>* %217, align 8, !tbaa !30, !alias.scope !47, !noalias !44
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
  call void @llvm.experimental.noalias.scope.decl(metadata !44) #14
  call void @llvm.experimental.noalias.scope.decl(metadata !47) #14
  %226 = getelementptr inbounds %"class.std::thread", %"class.std::thread"* %224, i64 0, i32 0, i32 0
  %227 = getelementptr inbounds %"class.std::thread", %"class.std::thread"* %225, i64 0, i32 0, i32 0
  %228 = load i64, i64* %227, align 8, !tbaa !30, !alias.scope !47, !noalias !44
  store i64 %228, i64* %226, align 8, !tbaa !30, !alias.scope !44, !noalias !47
  store i64 0, i64* %227, align 8, !tbaa !30, !alias.scope !47, !noalias !44
  %229 = getelementptr inbounds %"class.std::thread", %"class.std::thread"* %225, i64 1
  %230 = getelementptr inbounds %"class.std::thread", %"class.std::thread"* %224, i64 1
  %231 = icmp eq %"class.std::thread"* %229, %9
  br i1 %231, label %232, label %223, !llvm.loop !54

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
  store %"class.std::thread"* %37, %"class.std::thread"** %11, align 8, !tbaa !27
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
!25 = distinct !{!25, !26}
!26 = !{!"llvm.loop.mustprogress"}
!27 = !{!12, !6, i64 0}
!28 = distinct !{!28, !26}
!29 = !{i64 0, i64 8, !30}
!30 = !{!16, !16, i64 0}
!31 = !{!32}
!32 = distinct !{!32, !33, !"_ZSt19__relocate_object_aISt6threadS0_SaIS0_EEvPT_PT0_RT1_: argument 0"}
!33 = distinct !{!33, !"_ZSt19__relocate_object_aISt6threadS0_SaIS0_EEvPT_PT0_RT1_"}
!34 = !{!35}
!35 = distinct !{!35, !33, !"_ZSt19__relocate_object_aISt6threadS0_SaIS0_EEvPT_PT0_RT1_: argument 1"}
!36 = !{!37}
!37 = distinct !{!37, !33, !"_ZSt19__relocate_object_aISt6threadS0_SaIS0_EEvPT_PT0_RT1_: argument 0:It1"}
!38 = !{!39}
!39 = distinct !{!39, !33, !"_ZSt19__relocate_object_aISt6threadS0_SaIS0_EEvPT_PT0_RT1_: argument 1:It1"}
!40 = distinct !{!40, !26, !41}
!41 = !{!"llvm.loop.isvectorized", i32 1}
!42 = distinct !{!42, !26, !43, !41}
!43 = !{!"llvm.loop.unroll.runtime.disable"}
!44 = !{!45}
!45 = distinct !{!45, !46, !"_ZSt19__relocate_object_aISt6threadS0_SaIS0_EEvPT_PT0_RT1_: argument 0"}
!46 = distinct !{!46, !"_ZSt19__relocate_object_aISt6threadS0_SaIS0_EEvPT_PT0_RT1_"}
!47 = !{!48}
!48 = distinct !{!48, !46, !"_ZSt19__relocate_object_aISt6threadS0_SaIS0_EEvPT_PT0_RT1_: argument 1"}
!49 = !{!50}
!50 = distinct !{!50, !46, !"_ZSt19__relocate_object_aISt6threadS0_SaIS0_EEvPT_PT0_RT1_: argument 0:It1"}
!51 = !{!52}
!52 = distinct !{!52, !46, !"_ZSt19__relocate_object_aISt6threadS0_SaIS0_EEvPT_PT0_RT1_: argument 1:It1"}
!53 = distinct !{!53, !26, !41}
!54 = distinct !{!54, !26, !43, !41}
