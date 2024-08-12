; ModuleID = 'cmmcParallelFor.cpp'
source_filename = "cmmcParallelFor.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

%"struct.(anonymous namespace)::Worker" = type { i32, i8*, %"struct.std::atomic", %"struct.std::atomic", %"struct.std::atomic.0", %"struct.std::atomic.2", %"struct.std::atomic.2", %"class.(anonymous namespace)::Futex", %"class.(anonymous namespace)::Futex" }
%"struct.std::atomic" = type { %"struct.std::__atomic_base" }
%"struct.std::__atomic_base" = type { i32 }
%"struct.std::atomic.0" = type { %"struct.std::__atomic_base.1" }
%"struct.std::__atomic_base.1" = type { void (i32, i32)* }
%"struct.std::atomic.2" = type { %"struct.std::__atomic_base.3" }
%"struct.std::__atomic_base.3" = type { i32 }
%"class.(anonymous namespace)::Futex" = type { %"struct.std::atomic" }
%struct.ParallelForEntry = type { void (i32, i32)*, i32, i8, i32, [3 x i64], i32 }
%struct.cpu_set_t = type { [16 x i64] }
%class.anon = type { void (i32, i32)**, i32*, i32*, i32* }
%struct.timespec = type { i64, i64 }
%"struct.std::array" = type { [4 x i8] }

$_ZNSt13__atomic_baseIjEaSEj = comdat any

$_ZNKSt13__atomic_baseIjEcvjEv = comdat any

$_ZNKSt6atomicIPFviiEE4loadESt12memory_order = comdat any

$_ZStanSt12memory_orderSt23__memory_order_modifier = comdat any

$__clang_call_terminate = comdat any

$_ZSt23__cmpexch_failure_orderSt12memory_order = comdat any

$_ZStorSt12memory_orderSt23__memory_order_modifier = comdat any

$_ZSt24__cmpexch_failure_order2St12memory_order = comdat any

$_ZNSt14numeric_limitsIlE3maxEv = comdat any

$_ZNSt14numeric_limitsIjE3maxEv = comdat any

$_ZSt3minIiERKT_S2_S2_ = comdat any

$_ZNSt6atomicIPFviiEEaSES1_ = comdat any

$_ZNSt13__atomic_baseIiEaSEi = comdat any

$_ZNSt5arrayIbLm4EEixEm = comdat any

$_ZNSt13__atomic_baseIPFviiEEaSES1_ = comdat any

$_ZNSt14__array_traitsIbLm4EE6_S_refERA4_Kbm = comdat any

@_ZN12_GLOBAL__N_17workersE = internal global [4 x %"struct.(anonymous namespace)::Worker"] zeroinitializer, align 16
@_ZL9lookupPtr = internal global i32 0, align 4
@_ZL13parallelCache = internal global [16 x %struct.ParallelForEntry] zeroinitializer, align 16
@llvm.global_ctors = appending global [1 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 65535, void ()* @cmmcInitRuntime, i8* null }]
@llvm.global_dtors = appending global [1 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 65535, void ()* @cmmcUninitRuntime, i8* null }]

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define dso_local void @cmmcInitRuntime() #0 {
  %1 = alloca i32, align 4
  %2 = alloca %"struct.(anonymous namespace)::Worker"*, align 8
  store i32 0, i32* %1, align 4
  br label %3

3:                                                ; preds = %31, %0
  %4 = load i32, i32* %1, align 4
  %5 = icmp ult i32 %4, 4
  br i1 %5, label %6, label %34

6:                                                ; preds = %3
  %7 = load i32, i32* %1, align 4
  %8 = zext i32 %7 to i64
  %9 = getelementptr inbounds [4 x %"struct.(anonymous namespace)::Worker"], [4 x %"struct.(anonymous namespace)::Worker"]* @_ZN12_GLOBAL__N_17workersE, i64 0, i64 %8
  store %"struct.(anonymous namespace)::Worker"* %9, %"struct.(anonymous namespace)::Worker"** %2, align 8
  %10 = load %"struct.(anonymous namespace)::Worker"*, %"struct.(anonymous namespace)::Worker"** %2, align 8
  %11 = getelementptr inbounds %"struct.(anonymous namespace)::Worker", %"struct.(anonymous namespace)::Worker"* %10, i32 0, i32 3
  %12 = bitcast %"struct.std::atomic"* %11 to %"struct.std::__atomic_base"*
  %13 = call noundef i32 @_ZNSt13__atomic_baseIjEaSEj(%"struct.std::__atomic_base"* noundef nonnull align 4 dereferenceable(4) %12, i32 noundef 1) #6
  %14 = call i8* @mmap(i8* noundef null, i64 noundef 1048576, i32 noundef 3, i32 noundef 131106, i32 noundef -1, i64 noundef 0) #6
  %15 = load %"struct.(anonymous namespace)::Worker"*, %"struct.(anonymous namespace)::Worker"** %2, align 8
  %16 = getelementptr inbounds %"struct.(anonymous namespace)::Worker", %"struct.(anonymous namespace)::Worker"* %15, i32 0, i32 1
  store i8* %14, i8** %16, align 8
  %17 = load i32, i32* %1, align 4
  %18 = load %"struct.(anonymous namespace)::Worker"*, %"struct.(anonymous namespace)::Worker"** %2, align 8
  %19 = getelementptr inbounds %"struct.(anonymous namespace)::Worker", %"struct.(anonymous namespace)::Worker"* %18, i32 0, i32 2
  %20 = bitcast %"struct.std::atomic"* %19 to %"struct.std::__atomic_base"*
  %21 = call noundef i32 @_ZNSt13__atomic_baseIjEaSEj(%"struct.std::__atomic_base"* noundef nonnull align 4 dereferenceable(4) %20, i32 noundef %17) #6
  %22 = load %"struct.(anonymous namespace)::Worker"*, %"struct.(anonymous namespace)::Worker"** %2, align 8
  %23 = getelementptr inbounds %"struct.(anonymous namespace)::Worker", %"struct.(anonymous namespace)::Worker"* %22, i32 0, i32 1
  %24 = load i8*, i8** %23, align 8
  %25 = getelementptr inbounds i8, i8* %24, i64 1048576
  %26 = load %"struct.(anonymous namespace)::Worker"*, %"struct.(anonymous namespace)::Worker"** %2, align 8
  %27 = bitcast %"struct.(anonymous namespace)::Worker"* %26 to i8*
  %28 = call i32 (i32 (i8*)*, i8*, i32, i8*, ...) @clone(i32 (i8*)* noundef @_ZN12_GLOBAL__N_110cmmcWorkerEPv, i8* noundef %25, i32 noundef 331520, i8* noundef %27) #6
  %29 = load %"struct.(anonymous namespace)::Worker"*, %"struct.(anonymous namespace)::Worker"** %2, align 8
  %30 = getelementptr inbounds %"struct.(anonymous namespace)::Worker", %"struct.(anonymous namespace)::Worker"* %29, i32 0, i32 0
  store i32 %28, i32* %30, align 8
  br label %31

31:                                               ; preds = %6
  %32 = load i32, i32* %1, align 4
  %33 = add i32 %32, 1
  store i32 %33, i32* %1, align 4
  br label %3, !llvm.loop !6

34:                                               ; preds = %3
  ret void
}

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define linkonce_odr dso_local noundef i32 @_ZNSt13__atomic_baseIjEaSEj(%"struct.std::__atomic_base"* noundef nonnull align 4 dereferenceable(4) %0, i32 noundef %1) #0 comdat align 2 {
  %3 = alloca %"struct.std::__atomic_base"*, align 8
  %4 = alloca i32, align 4
  %5 = alloca i32, align 4
  %6 = alloca i32, align 4
  %7 = alloca i32, align 4
  %8 = alloca %"struct.std::__atomic_base"*, align 8
  %9 = alloca i32, align 4
  store %"struct.std::__atomic_base"* %0, %"struct.std::__atomic_base"** %8, align 8
  store i32 %1, i32* %9, align 4
  %10 = load %"struct.std::__atomic_base"*, %"struct.std::__atomic_base"** %8, align 8
  %11 = load i32, i32* %9, align 4
  store %"struct.std::__atomic_base"* %10, %"struct.std::__atomic_base"** %3, align 8
  store i32 %11, i32* %4, align 4
  store i32 5, i32* %5, align 4
  %12 = load %"struct.std::__atomic_base"*, %"struct.std::__atomic_base"** %3, align 8
  %13 = load i32, i32* %5, align 4
  %14 = call noundef i32 @_ZStanSt12memory_orderSt23__memory_order_modifier(i32 noundef %13, i32 noundef 65535) #6
  store i32 %14, i32* %6, align 4
  %15 = getelementptr inbounds %"struct.std::__atomic_base", %"struct.std::__atomic_base"* %12, i32 0, i32 0
  %16 = load i32, i32* %5, align 4
  %17 = load i32, i32* %4, align 4
  store i32 %17, i32* %7, align 4
  switch i32 %16, label %18 [
    i32 3, label %20
    i32 5, label %22
  ]

18:                                               ; preds = %2
  %19 = load i32, i32* %7, align 4
  store atomic i32 %19, i32* %15 monotonic, align 4
  br label %24

20:                                               ; preds = %2
  %21 = load i32, i32* %7, align 4
  store atomic i32 %21, i32* %15 release, align 4
  br label %24

22:                                               ; preds = %2
  %23 = load i32, i32* %7, align 4
  store atomic i32 %23, i32* %15 seq_cst, align 4
  br label %24

24:                                               ; preds = %18, %20, %22
  %25 = load i32, i32* %9, align 4
  ret i32 %25
}

; Function Attrs: nounwind
declare i8* @mmap(i8* noundef, i64 noundef, i32 noundef, i32 noundef, i32 noundef, i64 noundef) #1

; Function Attrs: nounwind
declare i32 @clone(i32 (i8*)* noundef, i8* noundef, i32 noundef, i8* noundef, ...) #1

; Function Attrs: mustprogress noinline optnone uwtable
define internal noundef i32 @_ZN12_GLOBAL__N_110cmmcWorkerEPv(i8* noundef %0) #2 {
  %2 = alloca %"struct.std::__atomic_base.3"*, align 8
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  %5 = alloca i32, align 4
  %6 = alloca %"struct.std::__atomic_base.3"*, align 8
  %7 = alloca i32, align 4
  %8 = alloca i32, align 4
  %9 = alloca i32, align 4
  %10 = alloca i32, align 4
  %11 = alloca i32, align 4
  %12 = alloca i8*, align 8
  %13 = alloca %"struct.(anonymous namespace)::Worker"*, align 8
  %14 = alloca %struct.cpu_set_t, align 8
  %15 = alloca i64, align 8
  %16 = alloca i64, align 8
  %17 = alloca i32, align 4
  store i8* %0, i8** %12, align 8
  %18 = load i8*, i8** %12, align 8
  %19 = bitcast i8* %18 to %"struct.(anonymous namespace)::Worker"*
  store %"struct.(anonymous namespace)::Worker"* %19, %"struct.(anonymous namespace)::Worker"** %13, align 8
  %20 = load %"struct.(anonymous namespace)::Worker"*, %"struct.(anonymous namespace)::Worker"** %13, align 8
  %21 = getelementptr inbounds %"struct.(anonymous namespace)::Worker", %"struct.(anonymous namespace)::Worker"* %20, i32 0, i32 2
  %22 = bitcast %"struct.std::atomic"* %21 to %"struct.std::__atomic_base"*
  %23 = call noundef i32 @_ZNKSt13__atomic_baseIjEcvjEv(%"struct.std::__atomic_base"* noundef nonnull align 4 dereferenceable(4) %22) #6
  %24 = zext i32 %23 to i64
  store i64 %24, i64* %15, align 8
  %25 = load i64, i64* %15, align 8
  %26 = udiv i64 %25, 8
  %27 = icmp ult i64 %26, 128
  br i1 %27, label %28, label %39

28:                                               ; preds = %1
  %29 = load i64, i64* %15, align 8
  %30 = urem i64 %29, 64
  %31 = shl i64 1, %30
  %32 = getelementptr inbounds %struct.cpu_set_t, %struct.cpu_set_t* %14, i32 0, i32 0
  %33 = getelementptr inbounds [16 x i64], [16 x i64]* %32, i64 0, i64 0
  %34 = load i64, i64* %15, align 8
  %35 = udiv i64 %34, 64
  %36 = getelementptr inbounds i64, i64* %33, i64 %35
  %37 = load i64, i64* %36, align 8
  %38 = or i64 %37, %31
  store i64 %38, i64* %36, align 8
  br label %40

39:                                               ; preds = %1
  br label %40

40:                                               ; preds = %39, %28
  %41 = phi i64 [ %38, %28 ], [ 0, %39 ]
  store i64 %41, i64* %16, align 8
  %42 = load i64, i64* %16, align 8
  %43 = call i64 (i64, ...) @syscall(i64 noundef 186) #6
  %44 = trunc i64 %43 to i32
  store i32 %44, i32* %17, align 4
  %45 = load i32, i32* %17, align 4
  %46 = call i32 @sched_setaffinity(i32 noundef %45, i64 noundef 128, %struct.cpu_set_t* noundef %14) #6
  br label %47

47:                                               ; preds = %109, %40
  %48 = load %"struct.(anonymous namespace)::Worker"*, %"struct.(anonymous namespace)::Worker"** %13, align 8
  %49 = getelementptr inbounds %"struct.(anonymous namespace)::Worker", %"struct.(anonymous namespace)::Worker"* %48, i32 0, i32 3
  %50 = bitcast %"struct.std::atomic"* %49 to %"struct.std::__atomic_base"*
  %51 = call noundef i32 @_ZNKSt13__atomic_baseIjEcvjEv(%"struct.std::__atomic_base"* noundef nonnull align 4 dereferenceable(4) %50) #6
  %52 = icmp ne i32 %51, 0
  br i1 %52, label %53, label %112

53:                                               ; preds = %47
  %54 = load %"struct.(anonymous namespace)::Worker"*, %"struct.(anonymous namespace)::Worker"** %13, align 8
  %55 = getelementptr inbounds %"struct.(anonymous namespace)::Worker", %"struct.(anonymous namespace)::Worker"* %54, i32 0, i32 7
  call void @_ZN12_GLOBAL__N_15Futex4waitEv(%"class.(anonymous namespace)::Futex"* noundef nonnull align 4 dereferenceable(4) %55)
  %56 = load %"struct.(anonymous namespace)::Worker"*, %"struct.(anonymous namespace)::Worker"** %13, align 8
  %57 = getelementptr inbounds %"struct.(anonymous namespace)::Worker", %"struct.(anonymous namespace)::Worker"* %56, i32 0, i32 3
  %58 = bitcast %"struct.std::atomic"* %57 to %"struct.std::__atomic_base"*
  %59 = call noundef i32 @_ZNKSt13__atomic_baseIjEcvjEv(%"struct.std::__atomic_base"* noundef nonnull align 4 dereferenceable(4) %58) #6
  %60 = icmp ne i32 %59, 0
  br i1 %60, label %62, label %61

61:                                               ; preds = %53
  br label %112

62:                                               ; preds = %53
  store i32 5, i32* %10, align 4
  %63 = load i32, i32* %10, align 4
  switch i32 %63, label %68 [
    i32 1, label %64
    i32 2, label %64
    i32 3, label %65
    i32 4, label %66
    i32 5, label %67
  ]

64:                                               ; preds = %62, %62
  fence acquire
  br label %68

65:                                               ; preds = %62
  fence release
  br label %68

66:                                               ; preds = %62
  fence acq_rel
  br label %68

67:                                               ; preds = %62
  fence seq_cst
  br label %68

68:                                               ; preds = %62, %64, %65, %66, %67
  %69 = load %"struct.(anonymous namespace)::Worker"*, %"struct.(anonymous namespace)::Worker"** %13, align 8
  %70 = getelementptr inbounds %"struct.(anonymous namespace)::Worker", %"struct.(anonymous namespace)::Worker"* %69, i32 0, i32 4
  %71 = call noundef void (i32, i32)* @_ZNKSt6atomicIPFviiEE4loadESt12memory_order(%"struct.std::atomic.0"* noundef nonnull align 8 dereferenceable(8) %70, i32 noundef 5) #6
  %72 = load %"struct.(anonymous namespace)::Worker"*, %"struct.(anonymous namespace)::Worker"** %13, align 8
  %73 = getelementptr inbounds %"struct.(anonymous namespace)::Worker", %"struct.(anonymous namespace)::Worker"* %72, i32 0, i32 5
  %74 = bitcast %"struct.std::atomic.2"* %73 to %"struct.std::__atomic_base.3"*
  store %"struct.std::__atomic_base.3"* %74, %"struct.std::__atomic_base.3"** %2, align 8
  store i32 5, i32* %3, align 4
  %75 = load %"struct.std::__atomic_base.3"*, %"struct.std::__atomic_base.3"** %2, align 8
  %76 = load i32, i32* %3, align 4
  %77 = call noundef i32 @_ZStanSt12memory_orderSt23__memory_order_modifier(i32 noundef %76, i32 noundef 65535) #6
  store i32 %77, i32* %4, align 4
  %78 = getelementptr inbounds %"struct.std::__atomic_base.3", %"struct.std::__atomic_base.3"* %75, i32 0, i32 0
  %79 = load i32, i32* %3, align 4
  switch i32 %79, label %80 [
    i32 1, label %82
    i32 2, label %82
    i32 5, label %84
  ]

80:                                               ; preds = %68
  %81 = load atomic i32, i32* %78 monotonic, align 4
  store i32 %81, i32* %5, align 4
  br label %86

82:                                               ; preds = %68, %68
  %83 = load atomic i32, i32* %78 acquire, align 4
  store i32 %83, i32* %5, align 4
  br label %86

84:                                               ; preds = %68
  %85 = load atomic i32, i32* %78 seq_cst, align 4
  store i32 %85, i32* %5, align 4
  br label %86

86:                                               ; preds = %80, %82, %84
  %87 = load i32, i32* %5, align 4
  %88 = load %"struct.(anonymous namespace)::Worker"*, %"struct.(anonymous namespace)::Worker"** %13, align 8
  %89 = getelementptr inbounds %"struct.(anonymous namespace)::Worker", %"struct.(anonymous namespace)::Worker"* %88, i32 0, i32 6
  %90 = bitcast %"struct.std::atomic.2"* %89 to %"struct.std::__atomic_base.3"*
  store %"struct.std::__atomic_base.3"* %90, %"struct.std::__atomic_base.3"** %6, align 8
  store i32 5, i32* %7, align 4
  %91 = load %"struct.std::__atomic_base.3"*, %"struct.std::__atomic_base.3"** %6, align 8
  %92 = load i32, i32* %7, align 4
  %93 = call noundef i32 @_ZStanSt12memory_orderSt23__memory_order_modifier(i32 noundef %92, i32 noundef 65535) #6
  store i32 %93, i32* %8, align 4
  %94 = getelementptr inbounds %"struct.std::__atomic_base.3", %"struct.std::__atomic_base.3"* %91, i32 0, i32 0
  %95 = load i32, i32* %7, align 4
  switch i32 %95, label %96 [
    i32 1, label %98
    i32 2, label %98
    i32 5, label %100
  ]

96:                                               ; preds = %86
  %97 = load atomic i32, i32* %94 monotonic, align 4
  store i32 %97, i32* %9, align 4
  br label %102

98:                                               ; preds = %86, %86
  %99 = load atomic i32, i32* %94 acquire, align 4
  store i32 %99, i32* %9, align 4
  br label %102

100:                                              ; preds = %86
  %101 = load atomic i32, i32* %94 seq_cst, align 4
  store i32 %101, i32* %9, align 4
  br label %102

102:                                              ; preds = %96, %98, %100
  %103 = load i32, i32* %9, align 4
  call void %71(i32 noundef %87, i32 noundef %103)
  store i32 5, i32* %11, align 4
  %104 = load i32, i32* %11, align 4
  switch i32 %104, label %109 [
    i32 1, label %105
    i32 2, label %105
    i32 3, label %106
    i32 4, label %107
    i32 5, label %108
  ]

105:                                              ; preds = %102, %102
  fence acquire
  br label %109

106:                                              ; preds = %102
  fence release
  br label %109

107:                                              ; preds = %102
  fence acq_rel
  br label %109

108:                                              ; preds = %102
  fence seq_cst
  br label %109

109:                                              ; preds = %102, %105, %106, %107, %108
  %110 = load %"struct.(anonymous namespace)::Worker"*, %"struct.(anonymous namespace)::Worker"** %13, align 8
  %111 = getelementptr inbounds %"struct.(anonymous namespace)::Worker", %"struct.(anonymous namespace)::Worker"* %110, i32 0, i32 8
  call void @_ZN12_GLOBAL__N_15Futex4postEv(%"class.(anonymous namespace)::Futex"* noundef nonnull align 4 dereferenceable(4) %111)
  br label %47, !llvm.loop !8

112:                                              ; preds = %61, %47
  ret i32 0
}

; Function Attrs: mustprogress noinline optnone uwtable
define dso_local void @cmmcUninitRuntime() #2 {
  %1 = alloca [4 x %"struct.(anonymous namespace)::Worker"]*, align 8
  %2 = alloca %"struct.(anonymous namespace)::Worker"*, align 8
  %3 = alloca %"struct.(anonymous namespace)::Worker"*, align 8
  %4 = alloca %"struct.(anonymous namespace)::Worker"*, align 8
  store [4 x %"struct.(anonymous namespace)::Worker"]* @_ZN12_GLOBAL__N_17workersE, [4 x %"struct.(anonymous namespace)::Worker"]** %1, align 8
  store %"struct.(anonymous namespace)::Worker"* getelementptr inbounds ([4 x %"struct.(anonymous namespace)::Worker"], [4 x %"struct.(anonymous namespace)::Worker"]* @_ZN12_GLOBAL__N_17workersE, i64 0, i64 0), %"struct.(anonymous namespace)::Worker"** %2, align 8
  store %"struct.(anonymous namespace)::Worker"* getelementptr inbounds ([4 x %"struct.(anonymous namespace)::Worker"], [4 x %"struct.(anonymous namespace)::Worker"]* @_ZN12_GLOBAL__N_17workersE, i64 1, i64 0), %"struct.(anonymous namespace)::Worker"** %3, align 8
  br label %5

5:                                                ; preds = %21, %0
  %6 = load %"struct.(anonymous namespace)::Worker"*, %"struct.(anonymous namespace)::Worker"** %2, align 8
  %7 = load %"struct.(anonymous namespace)::Worker"*, %"struct.(anonymous namespace)::Worker"** %3, align 8
  %8 = icmp ne %"struct.(anonymous namespace)::Worker"* %6, %7
  br i1 %8, label %9, label %24

9:                                                ; preds = %5
  %10 = load %"struct.(anonymous namespace)::Worker"*, %"struct.(anonymous namespace)::Worker"** %2, align 8
  store %"struct.(anonymous namespace)::Worker"* %10, %"struct.(anonymous namespace)::Worker"** %4, align 8
  %11 = load %"struct.(anonymous namespace)::Worker"*, %"struct.(anonymous namespace)::Worker"** %4, align 8
  %12 = getelementptr inbounds %"struct.(anonymous namespace)::Worker", %"struct.(anonymous namespace)::Worker"* %11, i32 0, i32 3
  %13 = bitcast %"struct.std::atomic"* %12 to %"struct.std::__atomic_base"*
  %14 = call noundef i32 @_ZNSt13__atomic_baseIjEaSEj(%"struct.std::__atomic_base"* noundef nonnull align 4 dereferenceable(4) %13, i32 noundef 0) #6
  %15 = load %"struct.(anonymous namespace)::Worker"*, %"struct.(anonymous namespace)::Worker"** %4, align 8
  %16 = getelementptr inbounds %"struct.(anonymous namespace)::Worker", %"struct.(anonymous namespace)::Worker"* %15, i32 0, i32 7
  call void @_ZN12_GLOBAL__N_15Futex4postEv(%"class.(anonymous namespace)::Futex"* noundef nonnull align 4 dereferenceable(4) %16)
  %17 = load %"struct.(anonymous namespace)::Worker"*, %"struct.(anonymous namespace)::Worker"** %4, align 8
  %18 = getelementptr inbounds %"struct.(anonymous namespace)::Worker", %"struct.(anonymous namespace)::Worker"* %17, i32 0, i32 0
  %19 = load i32, i32* %18, align 8
  %20 = call i32 @waitpid(i32 noundef %19, i32* noundef null, i32 noundef 0)
  br label %21

21:                                               ; preds = %9
  %22 = load %"struct.(anonymous namespace)::Worker"*, %"struct.(anonymous namespace)::Worker"** %2, align 8
  %23 = getelementptr inbounds %"struct.(anonymous namespace)::Worker", %"struct.(anonymous namespace)::Worker"* %22, i32 1
  store %"struct.(anonymous namespace)::Worker"* %23, %"struct.(anonymous namespace)::Worker"** %2, align 8
  br label %5

24:                                               ; preds = %5
  ret void
}

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define internal void @_ZN12_GLOBAL__N_15Futex4postEv(%"class.(anonymous namespace)::Futex"* noundef nonnull align 4 dereferenceable(4) %0) #0 align 2 {
  %2 = alloca %"struct.std::__atomic_base"*, align 8
  %3 = alloca i32*, align 8
  %4 = alloca i32, align 4
  %5 = alloca i32, align 4
  %6 = alloca i32, align 4
  %7 = alloca i32, align 4
  %8 = alloca i8, align 1
  %9 = alloca %"struct.std::__atomic_base"*, align 8
  %10 = alloca i32*, align 8
  %11 = alloca i32, align 4
  %12 = alloca i32, align 4
  %13 = alloca %"class.(anonymous namespace)::Futex"*, align 8
  %14 = alloca i32, align 4
  store %"class.(anonymous namespace)::Futex"* %0, %"class.(anonymous namespace)::Futex"** %13, align 8
  %15 = load %"class.(anonymous namespace)::Futex"*, %"class.(anonymous namespace)::Futex"** %13, align 8
  store i32 0, i32* %14, align 4
  %16 = getelementptr inbounds %"class.(anonymous namespace)::Futex", %"class.(anonymous namespace)::Futex"* %15, i32 0, i32 0
  %17 = bitcast %"struct.std::atomic"* %16 to %"struct.std::__atomic_base"*
  store %"struct.std::__atomic_base"* %17, %"struct.std::__atomic_base"** %9, align 8
  store i32* %14, i32** %10, align 8
  store i32 1, i32* %11, align 4
  store i32 5, i32* %12, align 4
  %18 = load %"struct.std::__atomic_base"*, %"struct.std::__atomic_base"** %9, align 8
  %19 = load i32*, i32** %10, align 8
  %20 = load i32, i32* %11, align 4
  %21 = load i32, i32* %12, align 4
  %22 = load i32, i32* %12, align 4
  %23 = call noundef i32 @_ZSt23__cmpexch_failure_orderSt12memory_order(i32 noundef %22) #6
  store %"struct.std::__atomic_base"* %18, %"struct.std::__atomic_base"** %2, align 8
  store i32* %19, i32** %3, align 8
  store i32 %20, i32* %4, align 4
  store i32 %21, i32* %5, align 4
  store i32 %23, i32* %6, align 4
  %24 = load %"struct.std::__atomic_base"*, %"struct.std::__atomic_base"** %2, align 8
  %25 = getelementptr inbounds %"struct.std::__atomic_base", %"struct.std::__atomic_base"* %24, i32 0, i32 0
  %26 = load i32, i32* %5, align 4
  %27 = load i32*, i32** %3, align 8
  %28 = load i32, i32* %4, align 4
  store i32 %28, i32* %7, align 4
  %29 = load i32, i32* %6, align 4
  switch i32 %26, label %30 [
    i32 1, label %31
    i32 2, label %31
    i32 3, label %32
    i32 4, label %33
    i32 5, label %34
  ]

30:                                               ; preds = %1
  switch i32 %29, label %35 [
    i32 1, label %41
    i32 2, label %41
    i32 5, label %47
  ]

31:                                               ; preds = %1, %1
  switch i32 %29, label %63 [
    i32 1, label %69
    i32 2, label %69
    i32 5, label %75
  ]

32:                                               ; preds = %1
  switch i32 %29, label %91 [
    i32 1, label %97
    i32 2, label %97
    i32 5, label %103
  ]

33:                                               ; preds = %1
  switch i32 %29, label %119 [
    i32 1, label %125
    i32 2, label %125
    i32 5, label %131
  ]

34:                                               ; preds = %1
  switch i32 %29, label %147 [
    i32 1, label %153
    i32 2, label %153
    i32 5, label %159
  ]

35:                                               ; preds = %30
  %36 = load i32, i32* %27, align 4
  %37 = load i32, i32* %7, align 4
  %38 = cmpxchg i32* %25, i32 %36, i32 %37 monotonic monotonic, align 4
  %39 = extractvalue { i32, i1 } %38, 0
  %40 = extractvalue { i32, i1 } %38, 1
  br i1 %40, label %55, label %54

41:                                               ; preds = %30, %30
  %42 = load i32, i32* %27, align 4
  %43 = load i32, i32* %7, align 4
  %44 = cmpxchg i32* %25, i32 %42, i32 %43 monotonic acquire, align 4
  %45 = extractvalue { i32, i1 } %44, 0
  %46 = extractvalue { i32, i1 } %44, 1
  br i1 %46, label %58, label %57

47:                                               ; preds = %30
  %48 = load i32, i32* %27, align 4
  %49 = load i32, i32* %7, align 4
  %50 = cmpxchg i32* %25, i32 %48, i32 %49 monotonic seq_cst, align 4
  %51 = extractvalue { i32, i1 } %50, 0
  %52 = extractvalue { i32, i1 } %50, 1
  br i1 %52, label %61, label %60

53:                                               ; preds = %61, %58, %55
  br label %175

54:                                               ; preds = %35
  store i32 %39, i32* %27, align 4
  br label %55

55:                                               ; preds = %54, %35
  %56 = zext i1 %40 to i8
  store i8 %56, i8* %8, align 1
  br label %53

57:                                               ; preds = %41
  store i32 %45, i32* %27, align 4
  br label %58

58:                                               ; preds = %57, %41
  %59 = zext i1 %46 to i8
  store i8 %59, i8* %8, align 1
  br label %53

60:                                               ; preds = %47
  store i32 %51, i32* %27, align 4
  br label %61

61:                                               ; preds = %60, %47
  %62 = zext i1 %52 to i8
  store i8 %62, i8* %8, align 1
  br label %53

63:                                               ; preds = %31
  %64 = load i32, i32* %27, align 4
  %65 = load i32, i32* %7, align 4
  %66 = cmpxchg i32* %25, i32 %64, i32 %65 acquire monotonic, align 4
  %67 = extractvalue { i32, i1 } %66, 0
  %68 = extractvalue { i32, i1 } %66, 1
  br i1 %68, label %83, label %82

69:                                               ; preds = %31, %31
  %70 = load i32, i32* %27, align 4
  %71 = load i32, i32* %7, align 4
  %72 = cmpxchg i32* %25, i32 %70, i32 %71 acquire acquire, align 4
  %73 = extractvalue { i32, i1 } %72, 0
  %74 = extractvalue { i32, i1 } %72, 1
  br i1 %74, label %86, label %85

75:                                               ; preds = %31
  %76 = load i32, i32* %27, align 4
  %77 = load i32, i32* %7, align 4
  %78 = cmpxchg i32* %25, i32 %76, i32 %77 acquire seq_cst, align 4
  %79 = extractvalue { i32, i1 } %78, 0
  %80 = extractvalue { i32, i1 } %78, 1
  br i1 %80, label %89, label %88

81:                                               ; preds = %89, %86, %83
  br label %175

82:                                               ; preds = %63
  store i32 %67, i32* %27, align 4
  br label %83

83:                                               ; preds = %82, %63
  %84 = zext i1 %68 to i8
  store i8 %84, i8* %8, align 1
  br label %81

85:                                               ; preds = %69
  store i32 %73, i32* %27, align 4
  br label %86

86:                                               ; preds = %85, %69
  %87 = zext i1 %74 to i8
  store i8 %87, i8* %8, align 1
  br label %81

88:                                               ; preds = %75
  store i32 %79, i32* %27, align 4
  br label %89

89:                                               ; preds = %88, %75
  %90 = zext i1 %80 to i8
  store i8 %90, i8* %8, align 1
  br label %81

91:                                               ; preds = %32
  %92 = load i32, i32* %27, align 4
  %93 = load i32, i32* %7, align 4
  %94 = cmpxchg i32* %25, i32 %92, i32 %93 release monotonic, align 4
  %95 = extractvalue { i32, i1 } %94, 0
  %96 = extractvalue { i32, i1 } %94, 1
  br i1 %96, label %111, label %110

97:                                               ; preds = %32, %32
  %98 = load i32, i32* %27, align 4
  %99 = load i32, i32* %7, align 4
  %100 = cmpxchg i32* %25, i32 %98, i32 %99 release acquire, align 4
  %101 = extractvalue { i32, i1 } %100, 0
  %102 = extractvalue { i32, i1 } %100, 1
  br i1 %102, label %114, label %113

103:                                              ; preds = %32
  %104 = load i32, i32* %27, align 4
  %105 = load i32, i32* %7, align 4
  %106 = cmpxchg i32* %25, i32 %104, i32 %105 release seq_cst, align 4
  %107 = extractvalue { i32, i1 } %106, 0
  %108 = extractvalue { i32, i1 } %106, 1
  br i1 %108, label %117, label %116

109:                                              ; preds = %117, %114, %111
  br label %175

110:                                              ; preds = %91
  store i32 %95, i32* %27, align 4
  br label %111

111:                                              ; preds = %110, %91
  %112 = zext i1 %96 to i8
  store i8 %112, i8* %8, align 1
  br label %109

113:                                              ; preds = %97
  store i32 %101, i32* %27, align 4
  br label %114

114:                                              ; preds = %113, %97
  %115 = zext i1 %102 to i8
  store i8 %115, i8* %8, align 1
  br label %109

116:                                              ; preds = %103
  store i32 %107, i32* %27, align 4
  br label %117

117:                                              ; preds = %116, %103
  %118 = zext i1 %108 to i8
  store i8 %118, i8* %8, align 1
  br label %109

119:                                              ; preds = %33
  %120 = load i32, i32* %27, align 4
  %121 = load i32, i32* %7, align 4
  %122 = cmpxchg i32* %25, i32 %120, i32 %121 acq_rel monotonic, align 4
  %123 = extractvalue { i32, i1 } %122, 0
  %124 = extractvalue { i32, i1 } %122, 1
  br i1 %124, label %139, label %138

125:                                              ; preds = %33, %33
  %126 = load i32, i32* %27, align 4
  %127 = load i32, i32* %7, align 4
  %128 = cmpxchg i32* %25, i32 %126, i32 %127 acq_rel acquire, align 4
  %129 = extractvalue { i32, i1 } %128, 0
  %130 = extractvalue { i32, i1 } %128, 1
  br i1 %130, label %142, label %141

131:                                              ; preds = %33
  %132 = load i32, i32* %27, align 4
  %133 = load i32, i32* %7, align 4
  %134 = cmpxchg i32* %25, i32 %132, i32 %133 acq_rel seq_cst, align 4
  %135 = extractvalue { i32, i1 } %134, 0
  %136 = extractvalue { i32, i1 } %134, 1
  br i1 %136, label %145, label %144

137:                                              ; preds = %145, %142, %139
  br label %175

138:                                              ; preds = %119
  store i32 %123, i32* %27, align 4
  br label %139

139:                                              ; preds = %138, %119
  %140 = zext i1 %124 to i8
  store i8 %140, i8* %8, align 1
  br label %137

141:                                              ; preds = %125
  store i32 %129, i32* %27, align 4
  br label %142

142:                                              ; preds = %141, %125
  %143 = zext i1 %130 to i8
  store i8 %143, i8* %8, align 1
  br label %137

144:                                              ; preds = %131
  store i32 %135, i32* %27, align 4
  br label %145

145:                                              ; preds = %144, %131
  %146 = zext i1 %136 to i8
  store i8 %146, i8* %8, align 1
  br label %137

147:                                              ; preds = %34
  %148 = load i32, i32* %27, align 4
  %149 = load i32, i32* %7, align 4
  %150 = cmpxchg i32* %25, i32 %148, i32 %149 seq_cst monotonic, align 4
  %151 = extractvalue { i32, i1 } %150, 0
  %152 = extractvalue { i32, i1 } %150, 1
  br i1 %152, label %167, label %166

153:                                              ; preds = %34, %34
  %154 = load i32, i32* %27, align 4
  %155 = load i32, i32* %7, align 4
  %156 = cmpxchg i32* %25, i32 %154, i32 %155 seq_cst acquire, align 4
  %157 = extractvalue { i32, i1 } %156, 0
  %158 = extractvalue { i32, i1 } %156, 1
  br i1 %158, label %170, label %169

159:                                              ; preds = %34
  %160 = load i32, i32* %27, align 4
  %161 = load i32, i32* %7, align 4
  %162 = cmpxchg i32* %25, i32 %160, i32 %161 seq_cst seq_cst, align 4
  %163 = extractvalue { i32, i1 } %162, 0
  %164 = extractvalue { i32, i1 } %162, 1
  br i1 %164, label %173, label %172

165:                                              ; preds = %173, %170, %167
  br label %175

166:                                              ; preds = %147
  store i32 %151, i32* %27, align 4
  br label %167

167:                                              ; preds = %166, %147
  %168 = zext i1 %152 to i8
  store i8 %168, i8* %8, align 1
  br label %165

169:                                              ; preds = %153
  store i32 %157, i32* %27, align 4
  br label %170

170:                                              ; preds = %169, %153
  %171 = zext i1 %158 to i8
  store i8 %171, i8* %8, align 1
  br label %165

172:                                              ; preds = %159
  store i32 %163, i32* %27, align 4
  br label %173

173:                                              ; preds = %172, %159
  %174 = zext i1 %164 to i8
  store i8 %174, i8* %8, align 1
  br label %165

175:                                              ; preds = %53, %81, %109, %137, %165
  %176 = load i8, i8* %8, align 1
  %177 = trunc i8 %176 to i1
  br i1 %177, label %178, label %182

178:                                              ; preds = %175
  %179 = getelementptr inbounds %"class.(anonymous namespace)::Futex", %"class.(anonymous namespace)::Futex"* %15, i32 0, i32 0
  %180 = ptrtoint %"struct.std::atomic"* %179 to i64
  %181 = call i64 (i64, ...) @syscall(i64 noundef 202, i64 noundef %180, i32 noundef 1, i32 noundef 1, i8* null, i8* null, i32 noundef 0) #6
  br label %182

182:                                              ; preds = %178, %175
  ret void
}

declare i32 @waitpid(i32 noundef, i32* noundef, i32 noundef) #3

; Function Attrs: mustprogress noinline optnone uwtable
define dso_local void @ParallelFor(i32 noundef %0, i32 noundef %1, void (i32, i32)* noundef %2) #2 {
  %4 = alloca i32, align 4
  %5 = alloca i32, align 4
  %6 = alloca void (i32, i32)*, align 8
  %7 = alloca i32, align 4
  %8 = alloca i32, align 4
  %9 = alloca %class.anon, align 8
  %10 = alloca i8, align 1
  %11 = alloca i32, align 4
  %12 = alloca %struct.ParallelForEntry*, align 8
  %13 = alloca i64, align 8
  %14 = alloca i64, align 8
  %15 = alloca i64, align 8
  store i32 %0, i32* %4, align 4
  store i32 %1, i32* %5, align 4
  store void (i32, i32)* %2, void (i32, i32)** %6, align 8
  %16 = load i32, i32* %5, align 4
  %17 = load i32, i32* %4, align 4
  %18 = icmp sle i32 %16, %17
  br i1 %18, label %19, label %20

19:                                               ; preds = %3
  br label %60

20:                                               ; preds = %3
  %21 = load i32, i32* %5, align 4
  %22 = load i32, i32* %4, align 4
  %23 = sub nsw i32 %21, %22
  store i32 %23, i32* %7, align 4
  store i32 16, i32* %8, align 4
  %24 = load i32, i32* %7, align 4
  %25 = icmp ult i32 %24, 16
  br i1 %25, label %26, label %30

26:                                               ; preds = %20
  %27 = load void (i32, i32)*, void (i32, i32)** %6, align 8
  %28 = load i32, i32* %4, align 4
  %29 = load i32, i32* %5, align 4
  call void %27(i32 noundef %28, i32 noundef %29)
  br label %60

30:                                               ; preds = %20
  %31 = getelementptr inbounds %class.anon, %class.anon* %9, i32 0, i32 0
  store void (i32, i32)** %6, void (i32, i32)*** %31, align 8
  %32 = getelementptr inbounds %class.anon, %class.anon* %9, i32 0, i32 1
  store i32* %4, i32** %32, align 8
  %33 = getelementptr inbounds %class.anon, %class.anon* %9, i32 0, i32 2
  store i32* %5, i32** %33, align 8
  %34 = getelementptr inbounds %class.anon, %class.anon* %9, i32 0, i32 3
  store i32* %7, i32** %34, align 8
  %35 = load void (i32, i32)*, void (i32, i32)** %6, align 8
  %36 = load i32, i32* %7, align 4
  %37 = call noundef nonnull align 8 dereferenceable(56) %struct.ParallelForEntry* @_ZL21selectNumberOfThreadsPFviiEjRjRb(void (i32, i32)* noundef %35, i32 noundef %36, i32* noundef nonnull align 4 dereferenceable(4) %11, i8* noundef nonnull align 1 dereferenceable(1) %10)
  store %struct.ParallelForEntry* %37, %struct.ParallelForEntry** %12, align 8
  %38 = load i8, i8* %10, align 1
  %39 = trunc i8 %38 to i1
  br i1 %39, label %40, label %42

40:                                               ; preds = %30
  %41 = call noundef i64 @_ZL12getTimePointv()
  store i64 %41, i64* %13, align 8
  br label %42

42:                                               ; preds = %40, %30
  %43 = load i32, i32* %11, align 4
  %44 = shl i32 1, %43
  call void @"_ZZ11ParallelForENK3$_0clEj"(%class.anon* noundef nonnull align 8 dereferenceable(32) %9, i32 noundef %44)
  %45 = load i8, i8* %10, align 1
  %46 = trunc i8 %45 to i1
  br i1 %46, label %47, label %60

47:                                               ; preds = %42
  %48 = call noundef i64 @_ZL12getTimePointv()
  store i64 %48, i64* %14, align 8
  %49 = load i64, i64* %14, align 8
  %50 = load i64, i64* %13, align 8
  %51 = sub nsw i64 %49, %50
  store i64 %51, i64* %15, align 8
  %52 = load i64, i64* %15, align 8
  %53 = load %struct.ParallelForEntry*, %struct.ParallelForEntry** %12, align 8
  %54 = getelementptr inbounds %struct.ParallelForEntry, %struct.ParallelForEntry* %53, i32 0, i32 4
  %55 = load i32, i32* %11, align 4
  %56 = zext i32 %55 to i64
  %57 = getelementptr inbounds [3 x i64], [3 x i64]* %54, i64 0, i64 %56
  %58 = load i64, i64* %57, align 8
  %59 = add nsw i64 %58, %52
  store i64 %59, i64* %57, align 8
  br label %60

60:                                               ; preds = %19, %26, %47, %42
  ret void
}

; Function Attrs: mustprogress noinline optnone uwtable
define internal noundef nonnull align 8 dereferenceable(56) %struct.ParallelForEntry* @_ZL21selectNumberOfThreadsPFviiEjRjRb(void (i32, i32)* noundef %0, i32 noundef %1, i32* noundef nonnull align 4 dereferenceable(4) %2, i8* noundef nonnull align 1 dereferenceable(1) %3) #2 {
  %5 = alloca %struct.ParallelForEntry*, align 8
  %6 = alloca void (i32, i32)*, align 8
  %7 = alloca i32, align 4
  %8 = alloca i32*, align 8
  %9 = alloca i8*, align 8
  %10 = alloca %struct.ParallelForEntry*, align 8
  %11 = alloca i32, align 4
  %12 = alloca i64, align 8
  %13 = alloca i32, align 4
  store void (i32, i32)* %0, void (i32, i32)** %6, align 8
  store i32 %1, i32* %7, align 4
  store i32* %2, i32** %8, align 8
  store i8* %3, i8** %9, align 8
  %14 = load void (i32, i32)*, void (i32, i32)** %6, align 8
  %15 = load i32, i32* %7, align 4
  %16 = call noundef nonnull align 8 dereferenceable(56) %struct.ParallelForEntry* @_ZL11selectEntryPFviiEj(void (i32, i32)* noundef %14, i32 noundef %15)
  store %struct.ParallelForEntry* %16, %struct.ParallelForEntry** %10, align 8
  %17 = load %struct.ParallelForEntry*, %struct.ParallelForEntry** %10, align 8
  %18 = getelementptr inbounds %struct.ParallelForEntry, %struct.ParallelForEntry* %17, i32 0, i32 3
  %19 = load i32, i32* %18, align 8
  %20 = icmp ult i32 %19, 100
  br i1 %20, label %21, label %25

21:                                               ; preds = %4
  %22 = load i32*, i32** %8, align 8
  store i32 2, i32* %22, align 4
  %23 = load i8*, i8** %9, align 8
  store i8 0, i8* %23, align 1
  %24 = load %struct.ParallelForEntry*, %struct.ParallelForEntry** %10, align 8
  store %struct.ParallelForEntry* %24, %struct.ParallelForEntry** %5, align 8
  br label %81

25:                                               ; preds = %4
  %26 = load %struct.ParallelForEntry*, %struct.ParallelForEntry** %10, align 8
  %27 = getelementptr inbounds %struct.ParallelForEntry, %struct.ParallelForEntry* %26, i32 0, i32 3
  %28 = load i32, i32* %27, align 8
  %29 = icmp ult i32 %28, 160
  br i1 %29, label %30, label %39

30:                                               ; preds = %25
  %31 = load %struct.ParallelForEntry*, %struct.ParallelForEntry** %10, align 8
  %32 = getelementptr inbounds %struct.ParallelForEntry, %struct.ParallelForEntry* %31, i32 0, i32 3
  %33 = load i32, i32* %32, align 8
  %34 = sub i32 %33, 100
  %35 = udiv i32 %34, 20
  %36 = load i32*, i32** %8, align 8
  store i32 %35, i32* %36, align 4
  %37 = load i8*, i8** %9, align 8
  store i8 1, i8* %37, align 1
  %38 = load %struct.ParallelForEntry*, %struct.ParallelForEntry** %10, align 8
  store %struct.ParallelForEntry* %38, %struct.ParallelForEntry** %5, align 8
  br label %81

39:                                               ; preds = %25
  %40 = load %struct.ParallelForEntry*, %struct.ParallelForEntry** %10, align 8
  %41 = getelementptr inbounds %struct.ParallelForEntry, %struct.ParallelForEntry* %40, i32 0, i32 5
  %42 = load i32, i32* %41, align 8
  %43 = icmp ne i32 %42, 0
  br i1 %43, label %74, label %44

44:                                               ; preds = %39
  store i32 0, i32* %11, align 4
  %45 = call noundef i64 @_ZNSt14numeric_limitsIlE3maxEv() #6
  store i64 %45, i64* %12, align 8
  store i32 0, i32* %13, align 4
  br label %46

46:                                               ; preds = %67, %44
  %47 = load i32, i32* %13, align 4
  %48 = icmp ult i32 %47, 3
  br i1 %48, label %49, label %70

49:                                               ; preds = %46
  %50 = load %struct.ParallelForEntry*, %struct.ParallelForEntry** %10, align 8
  %51 = getelementptr inbounds %struct.ParallelForEntry, %struct.ParallelForEntry* %50, i32 0, i32 4
  %52 = load i32, i32* %13, align 4
  %53 = zext i32 %52 to i64
  %54 = getelementptr inbounds [3 x i64], [3 x i64]* %51, i64 0, i64 %53
  %55 = load i64, i64* %54, align 8
  %56 = load i64, i64* %12, align 8
  %57 = icmp slt i64 %55, %56
  br i1 %57, label %58, label %66

58:                                               ; preds = %49
  %59 = load i32, i32* %13, align 4
  store i32 %59, i32* %11, align 4
  %60 = load %struct.ParallelForEntry*, %struct.ParallelForEntry** %10, align 8
  %61 = getelementptr inbounds %struct.ParallelForEntry, %struct.ParallelForEntry* %60, i32 0, i32 4
  %62 = load i32, i32* %13, align 4
  %63 = zext i32 %62 to i64
  %64 = getelementptr inbounds [3 x i64], [3 x i64]* %61, i64 0, i64 %63
  %65 = load i64, i64* %64, align 8
  store i64 %65, i64* %12, align 8
  br label %66

66:                                               ; preds = %58, %49
  br label %67

67:                                               ; preds = %66
  %68 = load i32, i32* %13, align 4
  %69 = add i32 %68, 1
  store i32 %69, i32* %13, align 4
  br label %46, !llvm.loop !9

70:                                               ; preds = %46
  %71 = load i32, i32* %11, align 4
  %72 = load %struct.ParallelForEntry*, %struct.ParallelForEntry** %10, align 8
  %73 = getelementptr inbounds %struct.ParallelForEntry, %struct.ParallelForEntry* %72, i32 0, i32 5
  store i32 %71, i32* %73, align 8
  br label %74

74:                                               ; preds = %70, %39
  %75 = load %struct.ParallelForEntry*, %struct.ParallelForEntry** %10, align 8
  %76 = getelementptr inbounds %struct.ParallelForEntry, %struct.ParallelForEntry* %75, i32 0, i32 5
  %77 = load i32, i32* %76, align 8
  %78 = load i32*, i32** %8, align 8
  store i32 %77, i32* %78, align 4
  %79 = load i8*, i8** %9, align 8
  store i8 0, i8* %79, align 1
  %80 = load %struct.ParallelForEntry*, %struct.ParallelForEntry** %10, align 8
  store %struct.ParallelForEntry* %80, %struct.ParallelForEntry** %5, align 8
  br label %81

81:                                               ; preds = %74, %30, %21
  %82 = load %struct.ParallelForEntry*, %struct.ParallelForEntry** %5, align 8
  ret %struct.ParallelForEntry* %82
}

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define internal noundef i64 @_ZL12getTimePointv() #0 {
  %1 = alloca %struct.timespec, align 8
  %2 = call i32 @clock_gettime(i32 noundef 1, %struct.timespec* noundef %1) #6
  %3 = getelementptr inbounds %struct.timespec, %struct.timespec* %1, i32 0, i32 0
  %4 = load i64, i64* %3, align 8
  %5 = mul nsw i64 %4, 1000000000
  %6 = getelementptr inbounds %struct.timespec, %struct.timespec* %1, i32 0, i32 1
  %7 = load i64, i64* %6, align 8
  %8 = add nsw i64 %5, %7
  ret i64 %8
}

; Function Attrs: mustprogress noinline optnone uwtable
define internal void @"_ZZ11ParallelForENK3$_0clEj"(%class.anon* noundef nonnull align 8 dereferenceable(32) %0, i32 noundef %1) #2 align 2 {
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  %5 = alloca %class.anon*, align 8
  %6 = alloca i32, align 4
  %7 = alloca i32, align 4
  %8 = alloca i32, align 4
  %9 = alloca %"struct.std::array", align 1
  %10 = alloca i32, align 4
  %11 = alloca i32, align 4
  %12 = alloca i32, align 4
  %13 = alloca i32, align 4
  %14 = alloca %"struct.(anonymous namespace)::Worker"*, align 8
  %15 = alloca i32, align 4
  store %class.anon* %0, %class.anon** %5, align 8
  store i32 %1, i32* %6, align 4
  %16 = load %class.anon*, %class.anon** %5, align 8
  %17 = load i32, i32* %6, align 4
  %18 = icmp eq i32 %17, 1
  br i1 %18, label %19, label %29

19:                                               ; preds = %2
  %20 = getelementptr inbounds %class.anon, %class.anon* %16, i32 0, i32 0
  %21 = load void (i32, i32)**, void (i32, i32)*** %20, align 8
  %22 = load void (i32, i32)*, void (i32, i32)** %21, align 8
  %23 = getelementptr inbounds %class.anon, %class.anon* %16, i32 0, i32 1
  %24 = load i32*, i32** %23, align 8
  %25 = load i32, i32* %24, align 4
  %26 = getelementptr inbounds %class.anon, %class.anon* %16, i32 0, i32 2
  %27 = load i32*, i32** %26, align 8
  %28 = load i32, i32* %27, align 4
  call void %22(i32 noundef %25, i32 noundef %28)
  br label %133

29:                                               ; preds = %2
  store i32 5, i32* %3, align 4
  %30 = load i32, i32* %3, align 4
  switch i32 %30, label %35 [
    i32 1, label %31
    i32 2, label %31
    i32 3, label %32
    i32 4, label %33
    i32 5, label %34
  ]

31:                                               ; preds = %29, %29
  fence acquire
  br label %35

32:                                               ; preds = %29
  fence release
  br label %35

33:                                               ; preds = %29
  fence acq_rel
  br label %35

34:                                               ; preds = %29
  fence seq_cst
  br label %35

35:                                               ; preds = %29, %31, %32, %33, %34
  store i32 4, i32* %7, align 4
  %36 = getelementptr inbounds %class.anon, %class.anon* %16, i32 0, i32 3
  %37 = load i32*, i32** %36, align 8
  %38 = load i32, i32* %37, align 4
  %39 = load i32, i32* %6, align 4
  %40 = udiv i32 %38, %39
  %41 = add i32 %40, 4
  %42 = sub i32 %41, 1
  %43 = udiv i32 %42, 4
  %44 = mul i32 %43, 4
  store i32 %44, i32* %8, align 4
  %45 = bitcast %"struct.std::array"* %9 to i8*
  call void @llvm.memset.p0i8.i64(i8* align 1 %45, i8 0, i64 4, i1 false)
  store i32 0, i32* %10, align 4
  br label %46

46:                                               ; preds = %103, %35
  %47 = load i32, i32* %10, align 4
  %48 = load i32, i32* %6, align 4
  %49 = icmp slt i32 %47, %48
  br i1 %49, label %50, label %106

50:                                               ; preds = %46
  %51 = getelementptr inbounds %class.anon, %class.anon* %16, i32 0, i32 1
  %52 = load i32*, i32** %51, align 8
  %53 = load i32, i32* %52, align 4
  %54 = load i32, i32* %10, align 4
  %55 = load i32, i32* %8, align 4
  %56 = mul nsw i32 %54, %55
  %57 = add nsw i32 %53, %56
  store i32 %57, i32* %11, align 4
  %58 = load i32, i32* %11, align 4
  %59 = load i32, i32* %8, align 4
  %60 = add nsw i32 %58, %59
  store i32 %60, i32* %13, align 4
  %61 = getelementptr inbounds %class.anon, %class.anon* %16, i32 0, i32 2
  %62 = load i32*, i32** %61, align 8
  %63 = call noundef nonnull align 4 dereferenceable(4) i32* @_ZSt3minIiERKT_S2_S2_(i32* noundef nonnull align 4 dereferenceable(4) %13, i32* noundef nonnull align 4 dereferenceable(4) %62)
  %64 = load i32, i32* %63, align 4
  store i32 %64, i32* %12, align 4
  %65 = load i32, i32* %10, align 4
  %66 = load i32, i32* %6, align 4
  %67 = sub i32 %66, 1
  %68 = icmp eq i32 %65, %67
  br i1 %68, label %69, label %73

69:                                               ; preds = %50
  %70 = getelementptr inbounds %class.anon, %class.anon* %16, i32 0, i32 2
  %71 = load i32*, i32** %70, align 8
  %72 = load i32, i32* %71, align 4
  store i32 %72, i32* %12, align 4
  br label %73

73:                                               ; preds = %69, %50
  %74 = load i32, i32* %11, align 4
  %75 = load i32, i32* %12, align 4
  %76 = icmp sge i32 %74, %75
  br i1 %76, label %77, label %78

77:                                               ; preds = %73
  br label %103

78:                                               ; preds = %73
  %79 = load i32, i32* %10, align 4
  %80 = sext i32 %79 to i64
  %81 = getelementptr inbounds [4 x %"struct.(anonymous namespace)::Worker"], [4 x %"struct.(anonymous namespace)::Worker"]* @_ZN12_GLOBAL__N_17workersE, i64 0, i64 %80
  store %"struct.(anonymous namespace)::Worker"* %81, %"struct.(anonymous namespace)::Worker"** %14, align 8
  %82 = getelementptr inbounds %class.anon, %class.anon* %16, i32 0, i32 0
  %83 = load void (i32, i32)**, void (i32, i32)*** %82, align 8
  %84 = load void (i32, i32)*, void (i32, i32)** %83, align 8
  %85 = load %"struct.(anonymous namespace)::Worker"*, %"struct.(anonymous namespace)::Worker"** %14, align 8
  %86 = getelementptr inbounds %"struct.(anonymous namespace)::Worker", %"struct.(anonymous namespace)::Worker"* %85, i32 0, i32 4
  %87 = call noundef void (i32, i32)* @_ZNSt6atomicIPFviiEEaSES1_(%"struct.std::atomic.0"* noundef nonnull align 8 dereferenceable(8) %86, void (i32, i32)* noundef %84) #6
  %88 = load i32, i32* %11, align 4
  %89 = load %"struct.(anonymous namespace)::Worker"*, %"struct.(anonymous namespace)::Worker"** %14, align 8
  %90 = getelementptr inbounds %"struct.(anonymous namespace)::Worker", %"struct.(anonymous namespace)::Worker"* %89, i32 0, i32 5
  %91 = bitcast %"struct.std::atomic.2"* %90 to %"struct.std::__atomic_base.3"*
  %92 = call noundef i32 @_ZNSt13__atomic_baseIiEaSEi(%"struct.std::__atomic_base.3"* noundef nonnull align 4 dereferenceable(4) %91, i32 noundef %88) #6
  %93 = load i32, i32* %12, align 4
  %94 = load %"struct.(anonymous namespace)::Worker"*, %"struct.(anonymous namespace)::Worker"** %14, align 8
  %95 = getelementptr inbounds %"struct.(anonymous namespace)::Worker", %"struct.(anonymous namespace)::Worker"* %94, i32 0, i32 6
  %96 = bitcast %"struct.std::atomic.2"* %95 to %"struct.std::__atomic_base.3"*
  %97 = call noundef i32 @_ZNSt13__atomic_baseIiEaSEi(%"struct.std::__atomic_base.3"* noundef nonnull align 4 dereferenceable(4) %96, i32 noundef %93) #6
  %98 = load %"struct.(anonymous namespace)::Worker"*, %"struct.(anonymous namespace)::Worker"** %14, align 8
  %99 = getelementptr inbounds %"struct.(anonymous namespace)::Worker", %"struct.(anonymous namespace)::Worker"* %98, i32 0, i32 7
  call void @_ZN12_GLOBAL__N_15Futex4postEv(%"class.(anonymous namespace)::Futex"* noundef nonnull align 4 dereferenceable(4) %99)
  %100 = load i32, i32* %10, align 4
  %101 = sext i32 %100 to i64
  %102 = call noundef nonnull align 1 dereferenceable(1) i8* @_ZNSt5arrayIbLm4EEixEm(%"struct.std::array"* noundef nonnull align 1 dereferenceable(4) %9, i64 noundef %101) #6
  store i8 1, i8* %102, align 1
  br label %103

103:                                              ; preds = %78, %77
  %104 = load i32, i32* %10, align 4
  %105 = add nsw i32 %104, 1
  store i32 %105, i32* %10, align 4
  br label %46, !llvm.loop !10

106:                                              ; preds = %46
  store i32 0, i32* %15, align 4
  br label %107

107:                                              ; preds = %123, %106
  %108 = load i32, i32* %15, align 4
  %109 = load i32, i32* %6, align 4
  %110 = icmp ult i32 %108, %109
  br i1 %110, label %111, label %126

111:                                              ; preds = %107
  %112 = load i32, i32* %15, align 4
  %113 = zext i32 %112 to i64
  %114 = call noundef nonnull align 1 dereferenceable(1) i8* @_ZNSt5arrayIbLm4EEixEm(%"struct.std::array"* noundef nonnull align 1 dereferenceable(4) %9, i64 noundef %113) #6
  %115 = load i8, i8* %114, align 1
  %116 = trunc i8 %115 to i1
  br i1 %116, label %117, label %122

117:                                              ; preds = %111
  %118 = load i32, i32* %15, align 4
  %119 = zext i32 %118 to i64
  %120 = getelementptr inbounds [4 x %"struct.(anonymous namespace)::Worker"], [4 x %"struct.(anonymous namespace)::Worker"]* @_ZN12_GLOBAL__N_17workersE, i64 0, i64 %119
  %121 = getelementptr inbounds %"struct.(anonymous namespace)::Worker", %"struct.(anonymous namespace)::Worker"* %120, i32 0, i32 8
  call void @_ZN12_GLOBAL__N_15Futex4waitEv(%"class.(anonymous namespace)::Futex"* noundef nonnull align 4 dereferenceable(4) %121)
  br label %122

122:                                              ; preds = %117, %111
  br label %123

123:                                              ; preds = %122
  %124 = load i32, i32* %15, align 4
  %125 = add i32 %124, 1
  store i32 %125, i32* %15, align 4
  br label %107, !llvm.loop !11

126:                                              ; preds = %107
  store i32 5, i32* %4, align 4
  %127 = load i32, i32* %4, align 4
  switch i32 %127, label %132 [
    i32 1, label %128
    i32 2, label %128
    i32 3, label %129
    i32 4, label %130
    i32 5, label %131
  ]

128:                                              ; preds = %126, %126
  fence acquire
  br label %132

129:                                              ; preds = %126
  fence release
  br label %132

130:                                              ; preds = %126
  fence acq_rel
  br label %132

131:                                              ; preds = %126
  fence seq_cst
  br label %132

132:                                              ; preds = %126, %128, %129, %130, %131
  br label %133

133:                                              ; preds = %132, %19
  ret void
}

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define linkonce_odr dso_local noundef i32 @_ZNKSt13__atomic_baseIjEcvjEv(%"struct.std::__atomic_base"* noundef nonnull align 4 dereferenceable(4) %0) #0 comdat align 2 personality i32 (...)* @__gxx_personality_v0 {
  %2 = alloca %"struct.std::__atomic_base"*, align 8
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  %5 = alloca i32, align 4
  %6 = alloca %"struct.std::__atomic_base"*, align 8
  store %"struct.std::__atomic_base"* %0, %"struct.std::__atomic_base"** %6, align 8
  %7 = load %"struct.std::__atomic_base"*, %"struct.std::__atomic_base"** %6, align 8
  store %"struct.std::__atomic_base"* %7, %"struct.std::__atomic_base"** %2, align 8
  store i32 5, i32* %3, align 4
  %8 = load %"struct.std::__atomic_base"*, %"struct.std::__atomic_base"** %2, align 8
  %9 = load i32, i32* %3, align 4
  %10 = invoke noundef i32 @_ZStanSt12memory_orderSt23__memory_order_modifier(i32 noundef %9, i32 noundef 65535)
          to label %11 unwind label %20

11:                                               ; preds = %1
  store i32 %10, i32* %4, align 4
  %12 = getelementptr inbounds %"struct.std::__atomic_base", %"struct.std::__atomic_base"* %8, i32 0, i32 0
  %13 = load i32, i32* %3, align 4
  switch i32 %13, label %14 [
    i32 1, label %16
    i32 2, label %16
    i32 5, label %18
  ]

14:                                               ; preds = %11
  %15 = load atomic i32, i32* %12 monotonic, align 4
  store i32 %15, i32* %5, align 4
  br label %23

16:                                               ; preds = %11, %11
  %17 = load atomic i32, i32* %12 acquire, align 4
  store i32 %17, i32* %5, align 4
  br label %23

18:                                               ; preds = %11
  %19 = load atomic i32, i32* %12 seq_cst, align 4
  store i32 %19, i32* %5, align 4
  br label %23

20:                                               ; preds = %1
  %21 = landingpad { i8*, i32 }
          catch i8* null
  %22 = extractvalue { i8*, i32 } %21, 0
  call void @__clang_call_terminate(i8* %22) #7
  unreachable

23:                                               ; preds = %14, %16, %18
  %24 = load i32, i32* %5, align 4
  ret i32 %24
}

; Function Attrs: nounwind
declare i64 @syscall(i64 noundef, ...) #1

; Function Attrs: nounwind
declare i32 @sched_setaffinity(i32 noundef, i64 noundef, %struct.cpu_set_t* noundef) #1

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define internal void @_ZN12_GLOBAL__N_15Futex4waitEv(%"class.(anonymous namespace)::Futex"* noundef nonnull align 4 dereferenceable(4) %0) #0 align 2 {
  %2 = alloca %"struct.std::__atomic_base"*, align 8
  %3 = alloca i32*, align 8
  %4 = alloca i32, align 4
  %5 = alloca i32, align 4
  %6 = alloca i32, align 4
  %7 = alloca i32, align 4
  %8 = alloca i8, align 1
  %9 = alloca %"struct.std::__atomic_base"*, align 8
  %10 = alloca i32*, align 8
  %11 = alloca i32, align 4
  %12 = alloca i32, align 4
  %13 = alloca %"class.(anonymous namespace)::Futex"*, align 8
  %14 = alloca i32, align 4
  store %"class.(anonymous namespace)::Futex"* %0, %"class.(anonymous namespace)::Futex"** %13, align 8
  %15 = load %"class.(anonymous namespace)::Futex"*, %"class.(anonymous namespace)::Futex"** %13, align 8
  store i32 1, i32* %14, align 4
  br label %16

16:                                               ; preds = %180, %1
  %17 = getelementptr inbounds %"class.(anonymous namespace)::Futex", %"class.(anonymous namespace)::Futex"* %15, i32 0, i32 0
  %18 = bitcast %"struct.std::atomic"* %17 to %"struct.std::__atomic_base"*
  store %"struct.std::__atomic_base"* %18, %"struct.std::__atomic_base"** %9, align 8
  store i32* %14, i32** %10, align 8
  store i32 0, i32* %11, align 4
  store i32 5, i32* %12, align 4
  %19 = load %"struct.std::__atomic_base"*, %"struct.std::__atomic_base"** %9, align 8
  %20 = load i32*, i32** %10, align 8
  %21 = load i32, i32* %11, align 4
  %22 = load i32, i32* %12, align 4
  %23 = load i32, i32* %12, align 4
  %24 = call noundef i32 @_ZSt23__cmpexch_failure_orderSt12memory_order(i32 noundef %23) #6
  store %"struct.std::__atomic_base"* %19, %"struct.std::__atomic_base"** %2, align 8
  store i32* %20, i32** %3, align 8
  store i32 %21, i32* %4, align 4
  store i32 %22, i32* %5, align 4
  store i32 %24, i32* %6, align 4
  %25 = load %"struct.std::__atomic_base"*, %"struct.std::__atomic_base"** %2, align 8
  %26 = getelementptr inbounds %"struct.std::__atomic_base", %"struct.std::__atomic_base"* %25, i32 0, i32 0
  %27 = load i32, i32* %5, align 4
  %28 = load i32*, i32** %3, align 8
  %29 = load i32, i32* %4, align 4
  store i32 %29, i32* %7, align 4
  %30 = load i32, i32* %6, align 4
  switch i32 %27, label %31 [
    i32 1, label %32
    i32 2, label %32
    i32 3, label %33
    i32 4, label %34
    i32 5, label %35
  ]

31:                                               ; preds = %16
  switch i32 %30, label %36 [
    i32 1, label %42
    i32 2, label %42
    i32 5, label %48
  ]

32:                                               ; preds = %16, %16
  switch i32 %30, label %64 [
    i32 1, label %70
    i32 2, label %70
    i32 5, label %76
  ]

33:                                               ; preds = %16
  switch i32 %30, label %92 [
    i32 1, label %98
    i32 2, label %98
    i32 5, label %104
  ]

34:                                               ; preds = %16
  switch i32 %30, label %120 [
    i32 1, label %126
    i32 2, label %126
    i32 5, label %132
  ]

35:                                               ; preds = %16
  switch i32 %30, label %148 [
    i32 1, label %154
    i32 2, label %154
    i32 5, label %160
  ]

36:                                               ; preds = %31
  %37 = load i32, i32* %28, align 4
  %38 = load i32, i32* %7, align 4
  %39 = cmpxchg i32* %26, i32 %37, i32 %38 monotonic monotonic, align 4
  %40 = extractvalue { i32, i1 } %39, 0
  %41 = extractvalue { i32, i1 } %39, 1
  br i1 %41, label %56, label %55

42:                                               ; preds = %31, %31
  %43 = load i32, i32* %28, align 4
  %44 = load i32, i32* %7, align 4
  %45 = cmpxchg i32* %26, i32 %43, i32 %44 monotonic acquire, align 4
  %46 = extractvalue { i32, i1 } %45, 0
  %47 = extractvalue { i32, i1 } %45, 1
  br i1 %47, label %59, label %58

48:                                               ; preds = %31
  %49 = load i32, i32* %28, align 4
  %50 = load i32, i32* %7, align 4
  %51 = cmpxchg i32* %26, i32 %49, i32 %50 monotonic seq_cst, align 4
  %52 = extractvalue { i32, i1 } %51, 0
  %53 = extractvalue { i32, i1 } %51, 1
  br i1 %53, label %62, label %61

54:                                               ; preds = %62, %59, %56
  br label %176

55:                                               ; preds = %36
  store i32 %40, i32* %28, align 4
  br label %56

56:                                               ; preds = %55, %36
  %57 = zext i1 %41 to i8
  store i8 %57, i8* %8, align 1
  br label %54

58:                                               ; preds = %42
  store i32 %46, i32* %28, align 4
  br label %59

59:                                               ; preds = %58, %42
  %60 = zext i1 %47 to i8
  store i8 %60, i8* %8, align 1
  br label %54

61:                                               ; preds = %48
  store i32 %52, i32* %28, align 4
  br label %62

62:                                               ; preds = %61, %48
  %63 = zext i1 %53 to i8
  store i8 %63, i8* %8, align 1
  br label %54

64:                                               ; preds = %32
  %65 = load i32, i32* %28, align 4
  %66 = load i32, i32* %7, align 4
  %67 = cmpxchg i32* %26, i32 %65, i32 %66 acquire monotonic, align 4
  %68 = extractvalue { i32, i1 } %67, 0
  %69 = extractvalue { i32, i1 } %67, 1
  br i1 %69, label %84, label %83

70:                                               ; preds = %32, %32
  %71 = load i32, i32* %28, align 4
  %72 = load i32, i32* %7, align 4
  %73 = cmpxchg i32* %26, i32 %71, i32 %72 acquire acquire, align 4
  %74 = extractvalue { i32, i1 } %73, 0
  %75 = extractvalue { i32, i1 } %73, 1
  br i1 %75, label %87, label %86

76:                                               ; preds = %32
  %77 = load i32, i32* %28, align 4
  %78 = load i32, i32* %7, align 4
  %79 = cmpxchg i32* %26, i32 %77, i32 %78 acquire seq_cst, align 4
  %80 = extractvalue { i32, i1 } %79, 0
  %81 = extractvalue { i32, i1 } %79, 1
  br i1 %81, label %90, label %89

82:                                               ; preds = %90, %87, %84
  br label %176

83:                                               ; preds = %64
  store i32 %68, i32* %28, align 4
  br label %84

84:                                               ; preds = %83, %64
  %85 = zext i1 %69 to i8
  store i8 %85, i8* %8, align 1
  br label %82

86:                                               ; preds = %70
  store i32 %74, i32* %28, align 4
  br label %87

87:                                               ; preds = %86, %70
  %88 = zext i1 %75 to i8
  store i8 %88, i8* %8, align 1
  br label %82

89:                                               ; preds = %76
  store i32 %80, i32* %28, align 4
  br label %90

90:                                               ; preds = %89, %76
  %91 = zext i1 %81 to i8
  store i8 %91, i8* %8, align 1
  br label %82

92:                                               ; preds = %33
  %93 = load i32, i32* %28, align 4
  %94 = load i32, i32* %7, align 4
  %95 = cmpxchg i32* %26, i32 %93, i32 %94 release monotonic, align 4
  %96 = extractvalue { i32, i1 } %95, 0
  %97 = extractvalue { i32, i1 } %95, 1
  br i1 %97, label %112, label %111

98:                                               ; preds = %33, %33
  %99 = load i32, i32* %28, align 4
  %100 = load i32, i32* %7, align 4
  %101 = cmpxchg i32* %26, i32 %99, i32 %100 release acquire, align 4
  %102 = extractvalue { i32, i1 } %101, 0
  %103 = extractvalue { i32, i1 } %101, 1
  br i1 %103, label %115, label %114

104:                                              ; preds = %33
  %105 = load i32, i32* %28, align 4
  %106 = load i32, i32* %7, align 4
  %107 = cmpxchg i32* %26, i32 %105, i32 %106 release seq_cst, align 4
  %108 = extractvalue { i32, i1 } %107, 0
  %109 = extractvalue { i32, i1 } %107, 1
  br i1 %109, label %118, label %117

110:                                              ; preds = %118, %115, %112
  br label %176

111:                                              ; preds = %92
  store i32 %96, i32* %28, align 4
  br label %112

112:                                              ; preds = %111, %92
  %113 = zext i1 %97 to i8
  store i8 %113, i8* %8, align 1
  br label %110

114:                                              ; preds = %98
  store i32 %102, i32* %28, align 4
  br label %115

115:                                              ; preds = %114, %98
  %116 = zext i1 %103 to i8
  store i8 %116, i8* %8, align 1
  br label %110

117:                                              ; preds = %104
  store i32 %108, i32* %28, align 4
  br label %118

118:                                              ; preds = %117, %104
  %119 = zext i1 %109 to i8
  store i8 %119, i8* %8, align 1
  br label %110

120:                                              ; preds = %34
  %121 = load i32, i32* %28, align 4
  %122 = load i32, i32* %7, align 4
  %123 = cmpxchg i32* %26, i32 %121, i32 %122 acq_rel monotonic, align 4
  %124 = extractvalue { i32, i1 } %123, 0
  %125 = extractvalue { i32, i1 } %123, 1
  br i1 %125, label %140, label %139

126:                                              ; preds = %34, %34
  %127 = load i32, i32* %28, align 4
  %128 = load i32, i32* %7, align 4
  %129 = cmpxchg i32* %26, i32 %127, i32 %128 acq_rel acquire, align 4
  %130 = extractvalue { i32, i1 } %129, 0
  %131 = extractvalue { i32, i1 } %129, 1
  br i1 %131, label %143, label %142

132:                                              ; preds = %34
  %133 = load i32, i32* %28, align 4
  %134 = load i32, i32* %7, align 4
  %135 = cmpxchg i32* %26, i32 %133, i32 %134 acq_rel seq_cst, align 4
  %136 = extractvalue { i32, i1 } %135, 0
  %137 = extractvalue { i32, i1 } %135, 1
  br i1 %137, label %146, label %145

138:                                              ; preds = %146, %143, %140
  br label %176

139:                                              ; preds = %120
  store i32 %124, i32* %28, align 4
  br label %140

140:                                              ; preds = %139, %120
  %141 = zext i1 %125 to i8
  store i8 %141, i8* %8, align 1
  br label %138

142:                                              ; preds = %126
  store i32 %130, i32* %28, align 4
  br label %143

143:                                              ; preds = %142, %126
  %144 = zext i1 %131 to i8
  store i8 %144, i8* %8, align 1
  br label %138

145:                                              ; preds = %132
  store i32 %136, i32* %28, align 4
  br label %146

146:                                              ; preds = %145, %132
  %147 = zext i1 %137 to i8
  store i8 %147, i8* %8, align 1
  br label %138

148:                                              ; preds = %35
  %149 = load i32, i32* %28, align 4
  %150 = load i32, i32* %7, align 4
  %151 = cmpxchg i32* %26, i32 %149, i32 %150 seq_cst monotonic, align 4
  %152 = extractvalue { i32, i1 } %151, 0
  %153 = extractvalue { i32, i1 } %151, 1
  br i1 %153, label %168, label %167

154:                                              ; preds = %35, %35
  %155 = load i32, i32* %28, align 4
  %156 = load i32, i32* %7, align 4
  %157 = cmpxchg i32* %26, i32 %155, i32 %156 seq_cst acquire, align 4
  %158 = extractvalue { i32, i1 } %157, 0
  %159 = extractvalue { i32, i1 } %157, 1
  br i1 %159, label %171, label %170

160:                                              ; preds = %35
  %161 = load i32, i32* %28, align 4
  %162 = load i32, i32* %7, align 4
  %163 = cmpxchg i32* %26, i32 %161, i32 %162 seq_cst seq_cst, align 4
  %164 = extractvalue { i32, i1 } %163, 0
  %165 = extractvalue { i32, i1 } %163, 1
  br i1 %165, label %174, label %173

166:                                              ; preds = %174, %171, %168
  br label %176

167:                                              ; preds = %148
  store i32 %152, i32* %28, align 4
  br label %168

168:                                              ; preds = %167, %148
  %169 = zext i1 %153 to i8
  store i8 %169, i8* %8, align 1
  br label %166

170:                                              ; preds = %154
  store i32 %158, i32* %28, align 4
  br label %171

171:                                              ; preds = %170, %154
  %172 = zext i1 %159 to i8
  store i8 %172, i8* %8, align 1
  br label %166

173:                                              ; preds = %160
  store i32 %164, i32* %28, align 4
  br label %174

174:                                              ; preds = %173, %160
  %175 = zext i1 %165 to i8
  store i8 %175, i8* %8, align 1
  br label %166

176:                                              ; preds = %54, %82, %110, %138, %166
  %177 = load i8, i8* %8, align 1
  %178 = trunc i8 %177 to i1
  %179 = xor i1 %178, true
  br i1 %179, label %180, label %184

180:                                              ; preds = %176
  store i32 1, i32* %14, align 4
  %181 = getelementptr inbounds %"class.(anonymous namespace)::Futex", %"class.(anonymous namespace)::Futex"* %15, i32 0, i32 0
  %182 = ptrtoint %"struct.std::atomic"* %181 to i64
  %183 = call i64 (i64, ...) @syscall(i64 noundef 202, i64 noundef %182, i32 noundef 0, i32 noundef 0, i8* null, i8* null, i32 noundef 0) #6
  br label %16, !llvm.loop !12

184:                                              ; preds = %176
  ret void
}

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define linkonce_odr dso_local noundef void (i32, i32)* @_ZNKSt6atomicIPFviiEE4loadESt12memory_order(%"struct.std::atomic.0"* noundef nonnull align 8 dereferenceable(8) %0, i32 noundef %1) #0 comdat align 2 {
  %3 = alloca %"struct.std::__atomic_base.1"*, align 8
  %4 = alloca i32, align 4
  %5 = alloca i32, align 4
  %6 = alloca void (i32, i32)*, align 8
  %7 = alloca %"struct.std::atomic.0"*, align 8
  %8 = alloca i32, align 4
  store %"struct.std::atomic.0"* %0, %"struct.std::atomic.0"** %7, align 8
  store i32 %1, i32* %8, align 4
  %9 = load %"struct.std::atomic.0"*, %"struct.std::atomic.0"** %7, align 8
  %10 = getelementptr inbounds %"struct.std::atomic.0", %"struct.std::atomic.0"* %9, i32 0, i32 0
  %11 = load i32, i32* %8, align 4
  store %"struct.std::__atomic_base.1"* %10, %"struct.std::__atomic_base.1"** %3, align 8
  store i32 %11, i32* %4, align 4
  %12 = load %"struct.std::__atomic_base.1"*, %"struct.std::__atomic_base.1"** %3, align 8
  %13 = load i32, i32* %4, align 4
  %14 = call noundef i32 @_ZStanSt12memory_orderSt23__memory_order_modifier(i32 noundef %13, i32 noundef 65535) #6
  store i32 %14, i32* %5, align 4
  %15 = getelementptr inbounds %"struct.std::__atomic_base.1", %"struct.std::__atomic_base.1"* %12, i32 0, i32 0
  %16 = load i32, i32* %4, align 4
  %17 = bitcast void (i32, i32)** %15 to i64*
  %18 = bitcast void (i32, i32)** %6 to i64*
  switch i32 %16, label %19 [
    i32 1, label %21
    i32 2, label %21
    i32 5, label %23
  ]

19:                                               ; preds = %2
  %20 = load atomic i64, i64* %17 monotonic, align 8
  store i64 %20, i64* %18, align 8
  br label %25

21:                                               ; preds = %2, %2
  %22 = load atomic i64, i64* %17 acquire, align 8
  store i64 %22, i64* %18, align 8
  br label %25

23:                                               ; preds = %2
  %24 = load atomic i64, i64* %17 seq_cst, align 8
  store i64 %24, i64* %18, align 8
  br label %25

25:                                               ; preds = %19, %21, %23
  %26 = load void (i32, i32)*, void (i32, i32)** %6, align 8
  ret void (i32, i32)* %26
}

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define linkonce_odr dso_local noundef i32 @_ZStanSt12memory_orderSt23__memory_order_modifier(i32 noundef %0, i32 noundef %1) #0 comdat {
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  store i32 %0, i32* %3, align 4
  store i32 %1, i32* %4, align 4
  %5 = load i32, i32* %3, align 4
  %6 = load i32, i32* %4, align 4
  %7 = and i32 %5, %6
  ret i32 %7
}

declare i32 @__gxx_personality_v0(...)

; Function Attrs: noinline noreturn nounwind
define linkonce_odr hidden void @__clang_call_terminate(i8* %0) #4 comdat {
  %2 = call i8* @__cxa_begin_catch(i8* %0) #6
  call void @_ZSt9terminatev() #7
  unreachable
}

declare i8* @__cxa_begin_catch(i8*)

declare void @_ZSt9terminatev()

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define linkonce_odr dso_local noundef i32 @_ZSt23__cmpexch_failure_orderSt12memory_order(i32 noundef %0) #0 comdat personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
  %2 = alloca i32, align 4
  store i32 %0, i32* %2, align 4
  %3 = load i32, i32* %2, align 4
  %4 = call noundef i32 @_ZStanSt12memory_orderSt23__memory_order_modifier(i32 noundef %3, i32 noundef 65535)
  %5 = call noundef i32 @_ZSt24__cmpexch_failure_order2St12memory_order(i32 noundef %4) #6
  %6 = load i32, i32* %2, align 4
  %7 = call noundef i32 @_ZStanSt12memory_orderSt23__memory_order_modifier(i32 noundef %6, i32 noundef -65536)
  %8 = invoke noundef i32 @_ZStorSt12memory_orderSt23__memory_order_modifier(i32 noundef %5, i32 noundef %7)
          to label %9 unwind label %10

9:                                                ; preds = %1
  ret i32 %8

10:                                               ; preds = %1
  %11 = landingpad { i8*, i32 }
          catch i8* null
  %12 = extractvalue { i8*, i32 } %11, 0
  call void @__clang_call_terminate(i8* %12) #7
  unreachable
}

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define linkonce_odr dso_local noundef i32 @_ZStorSt12memory_orderSt23__memory_order_modifier(i32 noundef %0, i32 noundef %1) #0 comdat {
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  store i32 %0, i32* %3, align 4
  store i32 %1, i32* %4, align 4
  %5 = load i32, i32* %3, align 4
  %6 = load i32, i32* %4, align 4
  %7 = or i32 %5, %6
  ret i32 %7
}

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define linkonce_odr dso_local noundef i32 @_ZSt24__cmpexch_failure_order2St12memory_order(i32 noundef %0) #0 comdat {
  %2 = alloca i32, align 4
  store i32 %0, i32* %2, align 4
  %3 = load i32, i32* %2, align 4
  %4 = icmp eq i32 %3, 4
  br i1 %4, label %5, label %6

5:                                                ; preds = %1
  br label %14

6:                                                ; preds = %1
  %7 = load i32, i32* %2, align 4
  %8 = icmp eq i32 %7, 3
  br i1 %8, label %9, label %10

9:                                                ; preds = %6
  br label %12

10:                                               ; preds = %6
  %11 = load i32, i32* %2, align 4
  br label %12

12:                                               ; preds = %10, %9
  %13 = phi i32 [ 0, %9 ], [ %11, %10 ]
  br label %14

14:                                               ; preds = %12, %5
  %15 = phi i32 [ 2, %5 ], [ %13, %12 ]
  ret i32 %15
}

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define internal noundef nonnull align 8 dereferenceable(56) %struct.ParallelForEntry* @_ZL11selectEntryPFviiEj(void (i32, i32)* noundef %0, i32 noundef %1) #0 {
  %3 = alloca %struct.ParallelForEntry*, align 8
  %4 = alloca void (i32, i32)*, align 8
  %5 = alloca i32, align 4
  %6 = alloca i32, align 4
  %7 = alloca %struct.ParallelForEntry*, align 8
  %8 = alloca i32, align 4
  %9 = alloca %struct.ParallelForEntry*, align 8
  %10 = alloca i32, align 4
  %11 = alloca i32, align 4
  %12 = alloca i32, align 4
  %13 = alloca %struct.ParallelForEntry*, align 8
  %14 = alloca %struct.ParallelForEntry*, align 8
  store void (i32, i32)* %0, void (i32, i32)** %4, align 8
  store i32 %1, i32* %5, align 4
  store i32 0, i32* %6, align 4
  br label %15

15:                                               ; preds = %49, %2
  %16 = load i32, i32* %6, align 4
  %17 = icmp ult i32 %16, 16
  br i1 %17, label %18, label %54

18:                                               ; preds = %15
  %19 = load i32, i32* @_ZL9lookupPtr, align 4
  %20 = icmp eq i32 %19, 16
  br i1 %20, label %21, label %22

21:                                               ; preds = %18
  store i32 0, i32* @_ZL9lookupPtr, align 4
  br label %22

22:                                               ; preds = %21, %18
  %23 = load i32, i32* @_ZL9lookupPtr, align 4
  %24 = zext i32 %23 to i64
  %25 = getelementptr inbounds [16 x %struct.ParallelForEntry], [16 x %struct.ParallelForEntry]* @_ZL13parallelCache, i64 0, i64 %24
  store %struct.ParallelForEntry* %25, %struct.ParallelForEntry** %7, align 8
  %26 = load %struct.ParallelForEntry*, %struct.ParallelForEntry** %7, align 8
  %27 = getelementptr inbounds %struct.ParallelForEntry, %struct.ParallelForEntry* %26, i32 0, i32 2
  %28 = load i8, i8* %27, align 4
  %29 = trunc i8 %28 to i1
  br i1 %29, label %30, label %48

30:                                               ; preds = %22
  %31 = load %struct.ParallelForEntry*, %struct.ParallelForEntry** %7, align 8
  %32 = getelementptr inbounds %struct.ParallelForEntry, %struct.ParallelForEntry* %31, i32 0, i32 0
  %33 = load void (i32, i32)*, void (i32, i32)** %32, align 8
  %34 = load void (i32, i32)*, void (i32, i32)** %4, align 8
  %35 = icmp eq void (i32, i32)* %33, %34
  br i1 %35, label %36, label %48

36:                                               ; preds = %30
  %37 = load %struct.ParallelForEntry*, %struct.ParallelForEntry** %7, align 8
  %38 = getelementptr inbounds %struct.ParallelForEntry, %struct.ParallelForEntry* %37, i32 0, i32 1
  %39 = load i32, i32* %38, align 8
  %40 = load i32, i32* %5, align 4
  %41 = icmp eq i32 %39, %40
  br i1 %41, label %42, label %48

42:                                               ; preds = %36
  %43 = load %struct.ParallelForEntry*, %struct.ParallelForEntry** %7, align 8
  %44 = getelementptr inbounds %struct.ParallelForEntry, %struct.ParallelForEntry* %43, i32 0, i32 3
  %45 = load i32, i32* %44, align 8
  %46 = add i32 %45, 1
  store i32 %46, i32* %44, align 8
  %47 = load %struct.ParallelForEntry*, %struct.ParallelForEntry** %7, align 8
  store %struct.ParallelForEntry* %47, %struct.ParallelForEntry** %3, align 8
  br label %120

48:                                               ; preds = %36, %30, %22
  br label %49

49:                                               ; preds = %48
  %50 = load i32, i32* %6, align 4
  %51 = add i32 %50, 1
  store i32 %51, i32* %6, align 4
  %52 = load i32, i32* @_ZL9lookupPtr, align 4
  %53 = add i32 %52, 1
  store i32 %53, i32* @_ZL9lookupPtr, align 4
  br label %15, !llvm.loop !13

54:                                               ; preds = %15
  store i32 0, i32* %8, align 4
  br label %55

55:                                               ; preds = %80, %54
  %56 = load i32, i32* %8, align 4
  %57 = icmp ult i32 %56, 16
  br i1 %57, label %58, label %83

58:                                               ; preds = %55
  %59 = load i32, i32* %8, align 4
  %60 = zext i32 %59 to i64
  %61 = getelementptr inbounds [16 x %struct.ParallelForEntry], [16 x %struct.ParallelForEntry]* @_ZL13parallelCache, i64 0, i64 %60
  store %struct.ParallelForEntry* %61, %struct.ParallelForEntry** %9, align 8
  %62 = load %struct.ParallelForEntry*, %struct.ParallelForEntry** %9, align 8
  %63 = getelementptr inbounds %struct.ParallelForEntry, %struct.ParallelForEntry* %62, i32 0, i32 2
  %64 = load i8, i8* %63, align 4
  %65 = trunc i8 %64 to i1
  br i1 %65, label %79, label %66

66:                                               ; preds = %58
  %67 = load %struct.ParallelForEntry*, %struct.ParallelForEntry** %9, align 8
  %68 = getelementptr inbounds %struct.ParallelForEntry, %struct.ParallelForEntry* %67, i32 0, i32 2
  store i8 1, i8* %68, align 4
  %69 = load void (i32, i32)*, void (i32, i32)** %4, align 8
  %70 = load %struct.ParallelForEntry*, %struct.ParallelForEntry** %9, align 8
  %71 = getelementptr inbounds %struct.ParallelForEntry, %struct.ParallelForEntry* %70, i32 0, i32 0
  store void (i32, i32)* %69, void (i32, i32)** %71, align 8
  %72 = load i32, i32* %5, align 4
  %73 = load %struct.ParallelForEntry*, %struct.ParallelForEntry** %9, align 8
  %74 = getelementptr inbounds %struct.ParallelForEntry, %struct.ParallelForEntry* %73, i32 0, i32 1
  store i32 %72, i32* %74, align 8
  %75 = load %struct.ParallelForEntry*, %struct.ParallelForEntry** %9, align 8
  %76 = getelementptr inbounds %struct.ParallelForEntry, %struct.ParallelForEntry* %75, i32 0, i32 3
  store i32 1, i32* %76, align 8
  %77 = load i32, i32* %8, align 4
  store i32 %77, i32* @_ZL9lookupPtr, align 4
  %78 = load %struct.ParallelForEntry*, %struct.ParallelForEntry** %9, align 8
  store %struct.ParallelForEntry* %78, %struct.ParallelForEntry** %3, align 8
  br label %120

79:                                               ; preds = %58
  br label %80

80:                                               ; preds = %79
  %81 = load i32, i32* %8, align 4
  %82 = add i32 %81, 1
  store i32 %82, i32* %8, align 4
  br label %55, !llvm.loop !14

83:                                               ; preds = %55
  %84 = call noundef i32 @_ZNSt14numeric_limitsIjE3maxEv() #6
  store i32 %84, i32* %10, align 4
  store i32 0, i32* %11, align 4
  store i32 0, i32* %12, align 4
  br label %85

85:                                               ; preds = %103, %83
  %86 = load i32, i32* %12, align 4
  %87 = icmp ult i32 %86, 16
  br i1 %87, label %88, label %106

88:                                               ; preds = %85
  %89 = load i32, i32* %12, align 4
  %90 = zext i32 %89 to i64
  %91 = getelementptr inbounds [16 x %struct.ParallelForEntry], [16 x %struct.ParallelForEntry]* @_ZL13parallelCache, i64 0, i64 %90
  store %struct.ParallelForEntry* %91, %struct.ParallelForEntry** %13, align 8
  %92 = load %struct.ParallelForEntry*, %struct.ParallelForEntry** %13, align 8
  %93 = getelementptr inbounds %struct.ParallelForEntry, %struct.ParallelForEntry* %92, i32 0, i32 3
  %94 = load i32, i32* %93, align 8
  %95 = load i32, i32* %10, align 4
  %96 = icmp ult i32 %94, %95
  br i1 %96, label %97, label %102

97:                                               ; preds = %88
  %98 = load i32, i32* %12, align 4
  store i32 %98, i32* %11, align 4
  %99 = load %struct.ParallelForEntry*, %struct.ParallelForEntry** %13, align 8
  %100 = getelementptr inbounds %struct.ParallelForEntry, %struct.ParallelForEntry* %99, i32 0, i32 3
  %101 = load i32, i32* %100, align 8
  store i32 %101, i32* %10, align 4
  br label %102

102:                                              ; preds = %97, %88
  br label %103

103:                                              ; preds = %102
  %104 = load i32, i32* %12, align 4
  %105 = add i32 %104, 1
  store i32 %105, i32* %12, align 4
  br label %85, !llvm.loop !15

106:                                              ; preds = %85
  %107 = load i32, i32* %11, align 4
  %108 = zext i32 %107 to i64
  %109 = getelementptr inbounds [16 x %struct.ParallelForEntry], [16 x %struct.ParallelForEntry]* @_ZL13parallelCache, i64 0, i64 %108
  store %struct.ParallelForEntry* %109, %struct.ParallelForEntry** %14, align 8
  %110 = load void (i32, i32)*, void (i32, i32)** %4, align 8
  %111 = load %struct.ParallelForEntry*, %struct.ParallelForEntry** %14, align 8
  %112 = getelementptr inbounds %struct.ParallelForEntry, %struct.ParallelForEntry* %111, i32 0, i32 0
  store void (i32, i32)* %110, void (i32, i32)** %112, align 8
  %113 = load i32, i32* %5, align 4
  %114 = load %struct.ParallelForEntry*, %struct.ParallelForEntry** %14, align 8
  %115 = getelementptr inbounds %struct.ParallelForEntry, %struct.ParallelForEntry* %114, i32 0, i32 1
  store i32 %113, i32* %115, align 8
  %116 = load %struct.ParallelForEntry*, %struct.ParallelForEntry** %14, align 8
  %117 = getelementptr inbounds %struct.ParallelForEntry, %struct.ParallelForEntry* %116, i32 0, i32 3
  store i32 1, i32* %117, align 8
  %118 = load i32, i32* %11, align 4
  store i32 %118, i32* @_ZL9lookupPtr, align 4
  %119 = load %struct.ParallelForEntry*, %struct.ParallelForEntry** %14, align 8
  store %struct.ParallelForEntry* %119, %struct.ParallelForEntry** %3, align 8
  br label %120

120:                                              ; preds = %106, %66, %42
  %121 = load %struct.ParallelForEntry*, %struct.ParallelForEntry** %3, align 8
  ret %struct.ParallelForEntry* %121
}

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define linkonce_odr dso_local noundef i64 @_ZNSt14numeric_limitsIlE3maxEv() #0 comdat align 2 {
  ret i64 9223372036854775807
}

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define linkonce_odr dso_local noundef i32 @_ZNSt14numeric_limitsIjE3maxEv() #0 comdat align 2 {
  ret i32 -1
}

; Function Attrs: nounwind
declare i32 @clock_gettime(i32 noundef, %struct.timespec* noundef) #1

; Function Attrs: argmemonly nofree nounwind willreturn writeonly
declare void @llvm.memset.p0i8.i64(i8* nocapture writeonly, i8, i64, i1 immarg) #5

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define linkonce_odr dso_local noundef nonnull align 4 dereferenceable(4) i32* @_ZSt3minIiERKT_S2_S2_(i32* noundef nonnull align 4 dereferenceable(4) %0, i32* noundef nonnull align 4 dereferenceable(4) %1) #0 comdat {
  %3 = alloca i32*, align 8
  %4 = alloca i32*, align 8
  %5 = alloca i32*, align 8
  store i32* %0, i32** %4, align 8
  store i32* %1, i32** %5, align 8
  %6 = load i32*, i32** %5, align 8
  %7 = load i32, i32* %6, align 4
  %8 = load i32*, i32** %4, align 8
  %9 = load i32, i32* %8, align 4
  %10 = icmp slt i32 %7, %9
  br i1 %10, label %11, label %13

11:                                               ; preds = %2
  %12 = load i32*, i32** %5, align 8
  store i32* %12, i32** %3, align 8
  br label %15

13:                                               ; preds = %2
  %14 = load i32*, i32** %4, align 8
  store i32* %14, i32** %3, align 8
  br label %15

15:                                               ; preds = %13, %11
  %16 = load i32*, i32** %3, align 8
  ret i32* %16
}

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define linkonce_odr dso_local noundef void (i32, i32)* @_ZNSt6atomicIPFviiEEaSES1_(%"struct.std::atomic.0"* noundef nonnull align 8 dereferenceable(8) %0, void (i32, i32)* noundef %1) #0 comdat align 2 {
  %3 = alloca %"struct.std::atomic.0"*, align 8
  %4 = alloca void (i32, i32)*, align 8
  store %"struct.std::atomic.0"* %0, %"struct.std::atomic.0"** %3, align 8
  store void (i32, i32)* %1, void (i32, i32)** %4, align 8
  %5 = load %"struct.std::atomic.0"*, %"struct.std::atomic.0"** %3, align 8
  %6 = getelementptr inbounds %"struct.std::atomic.0", %"struct.std::atomic.0"* %5, i32 0, i32 0
  %7 = load void (i32, i32)*, void (i32, i32)** %4, align 8
  %8 = call noundef void (i32, i32)* @_ZNSt13__atomic_baseIPFviiEEaSES1_(%"struct.std::__atomic_base.1"* noundef nonnull align 8 dereferenceable(8) %6, void (i32, i32)* noundef %7) #6
  ret void (i32, i32)* %8
}

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define linkonce_odr dso_local noundef i32 @_ZNSt13__atomic_baseIiEaSEi(%"struct.std::__atomic_base.3"* noundef nonnull align 4 dereferenceable(4) %0, i32 noundef %1) #0 comdat align 2 {
  %3 = alloca %"struct.std::__atomic_base.3"*, align 8
  %4 = alloca i32, align 4
  %5 = alloca i32, align 4
  %6 = alloca i32, align 4
  %7 = alloca i32, align 4
  %8 = alloca %"struct.std::__atomic_base.3"*, align 8
  %9 = alloca i32, align 4
  store %"struct.std::__atomic_base.3"* %0, %"struct.std::__atomic_base.3"** %8, align 8
  store i32 %1, i32* %9, align 4
  %10 = load %"struct.std::__atomic_base.3"*, %"struct.std::__atomic_base.3"** %8, align 8
  %11 = load i32, i32* %9, align 4
  store %"struct.std::__atomic_base.3"* %10, %"struct.std::__atomic_base.3"** %3, align 8
  store i32 %11, i32* %4, align 4
  store i32 5, i32* %5, align 4
  %12 = load %"struct.std::__atomic_base.3"*, %"struct.std::__atomic_base.3"** %3, align 8
  %13 = load i32, i32* %5, align 4
  %14 = call noundef i32 @_ZStanSt12memory_orderSt23__memory_order_modifier(i32 noundef %13, i32 noundef 65535) #6
  store i32 %14, i32* %6, align 4
  %15 = getelementptr inbounds %"struct.std::__atomic_base.3", %"struct.std::__atomic_base.3"* %12, i32 0, i32 0
  %16 = load i32, i32* %5, align 4
  %17 = load i32, i32* %4, align 4
  store i32 %17, i32* %7, align 4
  switch i32 %16, label %18 [
    i32 3, label %20
    i32 5, label %22
  ]

18:                                               ; preds = %2
  %19 = load i32, i32* %7, align 4
  store atomic i32 %19, i32* %15 monotonic, align 4
  br label %24

20:                                               ; preds = %2
  %21 = load i32, i32* %7, align 4
  store atomic i32 %21, i32* %15 release, align 4
  br label %24

22:                                               ; preds = %2
  %23 = load i32, i32* %7, align 4
  store atomic i32 %23, i32* %15 seq_cst, align 4
  br label %24

24:                                               ; preds = %18, %20, %22
  %25 = load i32, i32* %9, align 4
  ret i32 %25
}

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define linkonce_odr dso_local noundef nonnull align 1 dereferenceable(1) i8* @_ZNSt5arrayIbLm4EEixEm(%"struct.std::array"* noundef nonnull align 1 dereferenceable(4) %0, i64 noundef %1) #0 comdat align 2 {
  %3 = alloca %"struct.std::array"*, align 8
  %4 = alloca i64, align 8
  store %"struct.std::array"* %0, %"struct.std::array"** %3, align 8
  store i64 %1, i64* %4, align 8
  %5 = load %"struct.std::array"*, %"struct.std::array"** %3, align 8
  %6 = getelementptr inbounds %"struct.std::array", %"struct.std::array"* %5, i32 0, i32 0
  %7 = load i64, i64* %4, align 8
  %8 = call noundef nonnull align 1 dereferenceable(1) i8* @_ZNSt14__array_traitsIbLm4EE6_S_refERA4_Kbm([4 x i8]* noundef nonnull align 1 dereferenceable(4) %6, i64 noundef %7) #6
  ret i8* %8
}

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define linkonce_odr dso_local noundef void (i32, i32)* @_ZNSt13__atomic_baseIPFviiEEaSES1_(%"struct.std::__atomic_base.1"* noundef nonnull align 8 dereferenceable(8) %0, void (i32, i32)* noundef %1) #0 comdat align 2 {
  %3 = alloca %"struct.std::__atomic_base.1"*, align 8
  %4 = alloca void (i32, i32)*, align 8
  %5 = alloca i32, align 4
  %6 = alloca i32, align 4
  %7 = alloca void (i32, i32)*, align 8
  %8 = alloca %"struct.std::__atomic_base.1"*, align 8
  %9 = alloca void (i32, i32)*, align 8
  store %"struct.std::__atomic_base.1"* %0, %"struct.std::__atomic_base.1"** %8, align 8
  store void (i32, i32)* %1, void (i32, i32)** %9, align 8
  %10 = load %"struct.std::__atomic_base.1"*, %"struct.std::__atomic_base.1"** %8, align 8
  %11 = load void (i32, i32)*, void (i32, i32)** %9, align 8
  store %"struct.std::__atomic_base.1"* %10, %"struct.std::__atomic_base.1"** %3, align 8
  store void (i32, i32)* %11, void (i32, i32)** %4, align 8
  store i32 5, i32* %5, align 4
  %12 = load %"struct.std::__atomic_base.1"*, %"struct.std::__atomic_base.1"** %3, align 8
  %13 = load i32, i32* %5, align 4
  %14 = call noundef i32 @_ZStanSt12memory_orderSt23__memory_order_modifier(i32 noundef %13, i32 noundef 65535) #6
  store i32 %14, i32* %6, align 4
  %15 = getelementptr inbounds %"struct.std::__atomic_base.1", %"struct.std::__atomic_base.1"* %12, i32 0, i32 0
  %16 = load i32, i32* %5, align 4
  %17 = load void (i32, i32)*, void (i32, i32)** %4, align 8
  store void (i32, i32)* %17, void (i32, i32)** %7, align 8
  %18 = bitcast void (i32, i32)** %15 to i64*
  %19 = bitcast void (i32, i32)** %7 to i64*
  switch i32 %16, label %20 [
    i32 3, label %22
    i32 5, label %24
  ]

20:                                               ; preds = %2
  %21 = load i64, i64* %19, align 8
  store atomic i64 %21, i64* %18 monotonic, align 8
  br label %26

22:                                               ; preds = %2
  %23 = load i64, i64* %19, align 8
  store atomic i64 %23, i64* %18 release, align 8
  br label %26

24:                                               ; preds = %2
  %25 = load i64, i64* %19, align 8
  store atomic i64 %25, i64* %18 seq_cst, align 8
  br label %26

26:                                               ; preds = %20, %22, %24
  %27 = load void (i32, i32)*, void (i32, i32)** %9, align 8
  ret void (i32, i32)* %27
}

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define linkonce_odr dso_local noundef nonnull align 1 dereferenceable(1) i8* @_ZNSt14__array_traitsIbLm4EE6_S_refERA4_Kbm([4 x i8]* noundef nonnull align 1 dereferenceable(4) %0, i64 noundef %1) #0 comdat align 2 {
  %3 = alloca [4 x i8]*, align 8
  %4 = alloca i64, align 8
  store [4 x i8]* %0, [4 x i8]** %3, align 8
  store i64 %1, i64* %4, align 8
  %5 = load [4 x i8]*, [4 x i8]** %3, align 8
  %6 = load i64, i64* %4, align 8
  %7 = getelementptr inbounds [4 x i8], [4 x i8]* %5, i64 0, i64 %6
  ret i8* %7
}

attributes #0 = { mustprogress noinline nounwind optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #2 = { mustprogress noinline optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #3 = { "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #4 = { noinline noreturn nounwind }
attributes #5 = { argmemonly nofree nounwind willreturn writeonly }
attributes #6 = { nounwind }
attributes #7 = { noreturn nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 1}
!4 = !{i32 7, !"frame-pointer", i32 2}
!5 = !{!"Debian clang version 14.0.6"}
!6 = distinct !{!6, !7}
!7 = !{!"llvm.loop.mustprogress"}
!8 = distinct !{!8, !7}
!9 = distinct !{!9, !7}
!10 = distinct !{!10, !7}
!11 = distinct !{!11, !7}
!12 = distinct !{!12, !7}
!13 = distinct !{!13, !7}
!14 = distinct !{!14, !7}
!15 = distinct !{!15, !7}
