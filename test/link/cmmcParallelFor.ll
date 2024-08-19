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
%struct.timespec = type { i64, i64 }
%"struct.std::array" = type { [4 x i8] }

@_ZN12_GLOBAL__N_17workersE = internal global [4 x %"struct.(anonymous namespace)::Worker"] zeroinitializer, align 16
@_ZL9lookupPtr = internal unnamed_addr global i32 0, align 4
@_ZL13parallelCache = internal unnamed_addr global [16 x %struct.ParallelForEntry] zeroinitializer, align 16
@llvm.global_ctors = appending global [1 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 65535, void ()* @cmmcInitRuntime, i8* null }]
@llvm.global_dtors = appending global [1 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 65535, void ()* @cmmcUninitRuntime, i8* null }]

; Function Attrs: mustprogress nounwind uwtable
define dso_local void @cmmcInitRuntime() #0 {
  store atomic i32 1, i32* getelementptr inbounds ([4 x %"struct.(anonymous namespace)::Worker"], [4 x %"struct.(anonymous namespace)::Worker"]* @_ZN12_GLOBAL__N_17workersE, i64 0, i64 0, i32 3, i32 0, i32 0) seq_cst, align 4
  %1 = tail call i8* @mmap(i8* noundef null, i64 noundef 1048576, i32 noundef 3, i32 noundef 131106, i32 noundef -1, i64 noundef 0) #5
  store i8* %1, i8** getelementptr inbounds ([4 x %"struct.(anonymous namespace)::Worker"], [4 x %"struct.(anonymous namespace)::Worker"]* @_ZN12_GLOBAL__N_17workersE, i64 0, i64 0, i32 1), align 8, !tbaa !5
  store atomic i32 0, i32* getelementptr inbounds ([4 x %"struct.(anonymous namespace)::Worker"], [4 x %"struct.(anonymous namespace)::Worker"]* @_ZN12_GLOBAL__N_17workersE, i64 0, i64 0, i32 2, i32 0, i32 0) seq_cst, align 16
  %2 = load i8*, i8** getelementptr inbounds ([4 x %"struct.(anonymous namespace)::Worker"], [4 x %"struct.(anonymous namespace)::Worker"]* @_ZN12_GLOBAL__N_17workersE, i64 0, i64 0, i32 1), align 8, !tbaa !5
  %3 = getelementptr inbounds i8, i8* %2, i64 1048576
  %4 = tail call i32 (i32 (i8*)*, i8*, i32, i8*, ...) @clone(i32 (i8*)* noundef nonnull @_ZN12_GLOBAL__N_110cmmcWorkerEPv, i8* noundef nonnull %3, i32 noundef 331520, i8* noundef nonnull bitcast ([4 x %"struct.(anonymous namespace)::Worker"]* @_ZN12_GLOBAL__N_17workersE to i8*)) #5
  store i32 %4, i32* getelementptr inbounds ([4 x %"struct.(anonymous namespace)::Worker"], [4 x %"struct.(anonymous namespace)::Worker"]* @_ZN12_GLOBAL__N_17workersE, i64 0, i64 0, i32 0), align 16, !tbaa !16
  store atomic i32 1, i32* getelementptr inbounds ([4 x %"struct.(anonymous namespace)::Worker"], [4 x %"struct.(anonymous namespace)::Worker"]* @_ZN12_GLOBAL__N_17workersE, i64 0, i64 1, i32 3, i32 0, i32 0) seq_cst, align 4
  %5 = tail call i8* @mmap(i8* noundef null, i64 noundef 1048576, i32 noundef 3, i32 noundef 131106, i32 noundef -1, i64 noundef 0) #5
  store i8* %5, i8** getelementptr inbounds ([4 x %"struct.(anonymous namespace)::Worker"], [4 x %"struct.(anonymous namespace)::Worker"]* @_ZN12_GLOBAL__N_17workersE, i64 0, i64 1, i32 1), align 8, !tbaa !5
  store atomic i32 1, i32* getelementptr inbounds ([4 x %"struct.(anonymous namespace)::Worker"], [4 x %"struct.(anonymous namespace)::Worker"]* @_ZN12_GLOBAL__N_17workersE, i64 0, i64 1, i32 2, i32 0, i32 0) seq_cst, align 16
  %6 = load i8*, i8** getelementptr inbounds ([4 x %"struct.(anonymous namespace)::Worker"], [4 x %"struct.(anonymous namespace)::Worker"]* @_ZN12_GLOBAL__N_17workersE, i64 0, i64 1, i32 1), align 8, !tbaa !5
  %7 = getelementptr inbounds i8, i8* %6, i64 1048576
  %8 = tail call i32 (i32 (i8*)*, i8*, i32, i8*, ...) @clone(i32 (i8*)* noundef nonnull @_ZN12_GLOBAL__N_110cmmcWorkerEPv, i8* noundef nonnull %7, i32 noundef 331520, i8* noundef nonnull bitcast (%"struct.(anonymous namespace)::Worker"* getelementptr inbounds ([4 x %"struct.(anonymous namespace)::Worker"], [4 x %"struct.(anonymous namespace)::Worker"]* @_ZN12_GLOBAL__N_17workersE, i64 0, i64 1) to i8*)) #5
  store i32 %8, i32* getelementptr inbounds ([4 x %"struct.(anonymous namespace)::Worker"], [4 x %"struct.(anonymous namespace)::Worker"]* @_ZN12_GLOBAL__N_17workersE, i64 0, i64 1, i32 0), align 16, !tbaa !16
  store atomic i32 1, i32* getelementptr inbounds ([4 x %"struct.(anonymous namespace)::Worker"], [4 x %"struct.(anonymous namespace)::Worker"]* @_ZN12_GLOBAL__N_17workersE, i64 0, i64 2, i32 3, i32 0, i32 0) seq_cst, align 4
  %9 = tail call i8* @mmap(i8* noundef null, i64 noundef 1048576, i32 noundef 3, i32 noundef 131106, i32 noundef -1, i64 noundef 0) #5
  store i8* %9, i8** getelementptr inbounds ([4 x %"struct.(anonymous namespace)::Worker"], [4 x %"struct.(anonymous namespace)::Worker"]* @_ZN12_GLOBAL__N_17workersE, i64 0, i64 2, i32 1), align 8, !tbaa !5
  store atomic i32 2, i32* getelementptr inbounds ([4 x %"struct.(anonymous namespace)::Worker"], [4 x %"struct.(anonymous namespace)::Worker"]* @_ZN12_GLOBAL__N_17workersE, i64 0, i64 2, i32 2, i32 0, i32 0) seq_cst, align 16
  %10 = load i8*, i8** getelementptr inbounds ([4 x %"struct.(anonymous namespace)::Worker"], [4 x %"struct.(anonymous namespace)::Worker"]* @_ZN12_GLOBAL__N_17workersE, i64 0, i64 2, i32 1), align 8, !tbaa !5
  %11 = getelementptr inbounds i8, i8* %10, i64 1048576
  %12 = tail call i32 (i32 (i8*)*, i8*, i32, i8*, ...) @clone(i32 (i8*)* noundef nonnull @_ZN12_GLOBAL__N_110cmmcWorkerEPv, i8* noundef nonnull %11, i32 noundef 331520, i8* noundef nonnull bitcast (%"struct.(anonymous namespace)::Worker"* getelementptr inbounds ([4 x %"struct.(anonymous namespace)::Worker"], [4 x %"struct.(anonymous namespace)::Worker"]* @_ZN12_GLOBAL__N_17workersE, i64 0, i64 2) to i8*)) #5
  store i32 %12, i32* getelementptr inbounds ([4 x %"struct.(anonymous namespace)::Worker"], [4 x %"struct.(anonymous namespace)::Worker"]* @_ZN12_GLOBAL__N_17workersE, i64 0, i64 2, i32 0), align 16, !tbaa !16
  store atomic i32 1, i32* getelementptr inbounds ([4 x %"struct.(anonymous namespace)::Worker"], [4 x %"struct.(anonymous namespace)::Worker"]* @_ZN12_GLOBAL__N_17workersE, i64 0, i64 3, i32 3, i32 0, i32 0) seq_cst, align 4
  %13 = tail call i8* @mmap(i8* noundef null, i64 noundef 1048576, i32 noundef 3, i32 noundef 131106, i32 noundef -1, i64 noundef 0) #5
  store i8* %13, i8** getelementptr inbounds ([4 x %"struct.(anonymous namespace)::Worker"], [4 x %"struct.(anonymous namespace)::Worker"]* @_ZN12_GLOBAL__N_17workersE, i64 0, i64 3, i32 1), align 8, !tbaa !5
  store atomic i32 3, i32* getelementptr inbounds ([4 x %"struct.(anonymous namespace)::Worker"], [4 x %"struct.(anonymous namespace)::Worker"]* @_ZN12_GLOBAL__N_17workersE, i64 0, i64 3, i32 2, i32 0, i32 0) seq_cst, align 16
  %14 = load i8*, i8** getelementptr inbounds ([4 x %"struct.(anonymous namespace)::Worker"], [4 x %"struct.(anonymous namespace)::Worker"]* @_ZN12_GLOBAL__N_17workersE, i64 0, i64 3, i32 1), align 8, !tbaa !5
  %15 = getelementptr inbounds i8, i8* %14, i64 1048576
  %16 = tail call i32 (i32 (i8*)*, i8*, i32, i8*, ...) @clone(i32 (i8*)* noundef nonnull @_ZN12_GLOBAL__N_110cmmcWorkerEPv, i8* noundef nonnull %15, i32 noundef 331520, i8* noundef nonnull bitcast (%"struct.(anonymous namespace)::Worker"* getelementptr inbounds ([4 x %"struct.(anonymous namespace)::Worker"], [4 x %"struct.(anonymous namespace)::Worker"]* @_ZN12_GLOBAL__N_17workersE, i64 0, i64 3) to i8*)) #5
  store i32 %16, i32* getelementptr inbounds ([4 x %"struct.(anonymous namespace)::Worker"], [4 x %"struct.(anonymous namespace)::Worker"]* @_ZN12_GLOBAL__N_17workersE, i64 0, i64 3, i32 0), align 16, !tbaa !16
  ret void
}

; Function Attrs: argmemonly mustprogress nofree nosync nounwind willreturn
declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture) #1

; Function Attrs: nounwind
declare i8* @mmap(i8* noundef, i64 noundef, i32 noundef, i32 noundef, i32 noundef, i64 noundef) local_unnamed_addr #2

; Function Attrs: nounwind
declare i32 @clone(i32 (i8*)* noundef, i8* noundef, i32 noundef, i8* noundef, ...) local_unnamed_addr #2

; Function Attrs: mustprogress uwtable
define internal noundef i32 @_ZN12_GLOBAL__N_110cmmcWorkerEPv(i8* noundef %0) #3 personality i32 (...)* @__gxx_personality_v0 {
  %2 = alloca %struct.cpu_set_t, align 8
  %3 = bitcast %struct.cpu_set_t* %2 to i8*
  call void @llvm.lifetime.start.p0i8(i64 128, i8* nonnull %3) #5
  %4 = getelementptr inbounds i8, i8* %0, i64 16
  %5 = bitcast i8* %4 to i32*
  %6 = load atomic i32, i32* %5 seq_cst, align 4
  %7 = zext i32 %6 to i64
  %8 = icmp ult i32 %6, 1024
  br i1 %8, label %9, label %16

9:                                                ; preds = %1
  %10 = and i64 %7, 63
  %11 = shl nuw i64 1, %10
  %12 = lshr i64 %7, 6
  %13 = getelementptr inbounds %struct.cpu_set_t, %struct.cpu_set_t* %2, i64 0, i32 0, i64 %12
  %14 = load i64, i64* %13, align 8, !tbaa !17
  %15 = or i64 %14, %11
  store i64 %15, i64* %13, align 8, !tbaa !17
  br label %16

16:                                               ; preds = %1, %9
  %17 = tail call i64 (i64, ...) @syscall(i64 noundef 186) #5
  %18 = trunc i64 %17 to i32
  %19 = call i32 @sched_setaffinity(i32 noundef %18, i64 noundef 128, %struct.cpu_set_t* noundef nonnull %2) #5
  call void @llvm.lifetime.end.p0i8(i64 128, i8* nonnull %3) #5
  %20 = getelementptr inbounds i8, i8* %0, i64 20
  %21 = bitcast i8* %20 to i32*
  %22 = load atomic i32, i32* %21 seq_cst, align 4
  %23 = icmp eq i32 %22, 0
  br i1 %23, label %59, label %24

24:                                               ; preds = %16
  %25 = getelementptr inbounds i8, i8* %0, i64 40
  %26 = bitcast i8* %25 to i32*
  %27 = ptrtoint i8* %25 to i64
  %28 = getelementptr inbounds i8, i8* %0, i64 24
  %29 = bitcast i8* %28 to i64*
  %30 = getelementptr inbounds i8, i8* %0, i64 32
  %31 = bitcast i8* %30 to i32*
  %32 = getelementptr inbounds i8, i8* %0, i64 36
  %33 = bitcast i8* %32 to i32*
  %34 = getelementptr inbounds i8, i8* %0, i64 44
  %35 = bitcast i8* %34 to i32*
  %36 = ptrtoint i8* %34 to i64
  br label %37

37:                                               ; preds = %24, %56
  %38 = cmpxchg i32* %26, i32 1, i32 0 seq_cst seq_cst, align 4
  %39 = extractvalue { i32, i1 } %38, 1
  br i1 %39, label %44, label %40

40:                                               ; preds = %37, %40
  %41 = call i64 (i64, ...) @syscall(i64 noundef 202, i64 noundef %27, i32 noundef 0, i32 noundef 0, i8* null, i8* null, i32 noundef 0) #5
  %42 = cmpxchg i32* %26, i32 1, i32 0 seq_cst seq_cst, align 4
  %43 = extractvalue { i32, i1 } %42, 1
  br i1 %43, label %44, label %40, !llvm.loop !19

44:                                               ; preds = %40, %37
  %45 = load atomic i32, i32* %21 seq_cst, align 4
  %46 = icmp eq i32 %45, 0
  br i1 %46, label %59, label %47

47:                                               ; preds = %44
  fence seq_cst
  %48 = load atomic i64, i64* %29 seq_cst, align 8
  %49 = inttoptr i64 %48 to void (i32, i32)*
  %50 = load atomic i32, i32* %31 seq_cst, align 4
  %51 = load atomic i32, i32* %33 seq_cst, align 4
  call void %49(i32 noundef %50, i32 noundef %51)
  fence seq_cst
  %52 = cmpxchg i32* %35, i32 0, i32 1 seq_cst seq_cst, align 4
  %53 = extractvalue { i32, i1 } %52, 1
  br i1 %53, label %54, label %56

54:                                               ; preds = %47
  %55 = call i64 (i64, ...) @syscall(i64 noundef 202, i64 noundef %36, i32 noundef 1, i32 noundef 1, i8* null, i8* null, i32 noundef 0) #5
  br label %56

56:                                               ; preds = %47, %54
  %57 = load atomic i32, i32* %21 seq_cst, align 4
  %58 = icmp eq i32 %57, 0
  br i1 %58, label %59, label %37, !llvm.loop !21

59:                                               ; preds = %56, %44, %16
  ret i32 0
}

; Function Attrs: argmemonly mustprogress nofree nosync nounwind willreturn
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture) #1

; Function Attrs: mustprogress uwtable
define dso_local void @cmmcUninitRuntime() #3 personality i32 (...)* @__gxx_personality_v0 {
  store atomic i32 0, i32* getelementptr inbounds ([4 x %"struct.(anonymous namespace)::Worker"], [4 x %"struct.(anonymous namespace)::Worker"]* @_ZN12_GLOBAL__N_17workersE, i64 0, i64 0, i32 3, i32 0, i32 0) seq_cst, align 4
  %1 = cmpxchg i32* getelementptr inbounds ([4 x %"struct.(anonymous namespace)::Worker"], [4 x %"struct.(anonymous namespace)::Worker"]* @_ZN12_GLOBAL__N_17workersE, i64 0, i64 0, i32 7, i32 0, i32 0, i32 0), i32 0, i32 1 seq_cst seq_cst, align 4
  %2 = extractvalue { i32, i1 } %1, 1
  br i1 %2, label %3, label %5

3:                                                ; preds = %0
  %4 = tail call i64 (i64, ...) @syscall(i64 noundef 202, i64 noundef ptrtoint (%"class.(anonymous namespace)::Futex"* getelementptr inbounds ([4 x %"struct.(anonymous namespace)::Worker"], [4 x %"struct.(anonymous namespace)::Worker"]* @_ZN12_GLOBAL__N_17workersE, i64 0, i64 0, i32 7) to i64), i32 noundef 1, i32 noundef 1, i8* null, i8* null, i32 noundef 0) #5
  br label %5

5:                                                ; preds = %0, %3
  %6 = load i32, i32* getelementptr inbounds ([4 x %"struct.(anonymous namespace)::Worker"], [4 x %"struct.(anonymous namespace)::Worker"]* @_ZN12_GLOBAL__N_17workersE, i64 0, i64 0, i32 0), align 16, !tbaa !16
  %7 = tail call i32 @waitpid(i32 noundef %6, i32* noundef null, i32 noundef 0)
  store atomic i32 0, i32* getelementptr inbounds ([4 x %"struct.(anonymous namespace)::Worker"], [4 x %"struct.(anonymous namespace)::Worker"]* @_ZN12_GLOBAL__N_17workersE, i64 0, i64 1, i32 3, i32 0, i32 0) seq_cst, align 4
  %8 = cmpxchg i32* getelementptr inbounds ([4 x %"struct.(anonymous namespace)::Worker"], [4 x %"struct.(anonymous namespace)::Worker"]* @_ZN12_GLOBAL__N_17workersE, i64 0, i64 1, i32 7, i32 0, i32 0, i32 0), i32 0, i32 1 seq_cst seq_cst, align 4
  %9 = extractvalue { i32, i1 } %8, 1
  br i1 %9, label %10, label %12

10:                                               ; preds = %5
  %11 = tail call i64 (i64, ...) @syscall(i64 noundef 202, i64 noundef ptrtoint (%"class.(anonymous namespace)::Futex"* getelementptr inbounds ([4 x %"struct.(anonymous namespace)::Worker"], [4 x %"struct.(anonymous namespace)::Worker"]* @_ZN12_GLOBAL__N_17workersE, i64 0, i64 1, i32 7) to i64), i32 noundef 1, i32 noundef 1, i8* null, i8* null, i32 noundef 0) #5
  br label %12

12:                                               ; preds = %10, %5
  %13 = load i32, i32* getelementptr inbounds ([4 x %"struct.(anonymous namespace)::Worker"], [4 x %"struct.(anonymous namespace)::Worker"]* @_ZN12_GLOBAL__N_17workersE, i64 0, i64 1, i32 0), align 16, !tbaa !16
  %14 = tail call i32 @waitpid(i32 noundef %13, i32* noundef null, i32 noundef 0)
  store atomic i32 0, i32* getelementptr inbounds ([4 x %"struct.(anonymous namespace)::Worker"], [4 x %"struct.(anonymous namespace)::Worker"]* @_ZN12_GLOBAL__N_17workersE, i64 0, i64 2, i32 3, i32 0, i32 0) seq_cst, align 4
  %15 = cmpxchg i32* getelementptr inbounds ([4 x %"struct.(anonymous namespace)::Worker"], [4 x %"struct.(anonymous namespace)::Worker"]* @_ZN12_GLOBAL__N_17workersE, i64 0, i64 2, i32 7, i32 0, i32 0, i32 0), i32 0, i32 1 seq_cst seq_cst, align 4
  %16 = extractvalue { i32, i1 } %15, 1
  br i1 %16, label %17, label %19

17:                                               ; preds = %12
  %18 = tail call i64 (i64, ...) @syscall(i64 noundef 202, i64 noundef ptrtoint (%"class.(anonymous namespace)::Futex"* getelementptr inbounds ([4 x %"struct.(anonymous namespace)::Worker"], [4 x %"struct.(anonymous namespace)::Worker"]* @_ZN12_GLOBAL__N_17workersE, i64 0, i64 2, i32 7) to i64), i32 noundef 1, i32 noundef 1, i8* null, i8* null, i32 noundef 0) #5
  br label %19

19:                                               ; preds = %17, %12
  %20 = load i32, i32* getelementptr inbounds ([4 x %"struct.(anonymous namespace)::Worker"], [4 x %"struct.(anonymous namespace)::Worker"]* @_ZN12_GLOBAL__N_17workersE, i64 0, i64 2, i32 0), align 16, !tbaa !16
  %21 = tail call i32 @waitpid(i32 noundef %20, i32* noundef null, i32 noundef 0)
  store atomic i32 0, i32* getelementptr inbounds ([4 x %"struct.(anonymous namespace)::Worker"], [4 x %"struct.(anonymous namespace)::Worker"]* @_ZN12_GLOBAL__N_17workersE, i64 0, i64 3, i32 3, i32 0, i32 0) seq_cst, align 4
  %22 = cmpxchg i32* getelementptr inbounds ([4 x %"struct.(anonymous namespace)::Worker"], [4 x %"struct.(anonymous namespace)::Worker"]* @_ZN12_GLOBAL__N_17workersE, i64 0, i64 3, i32 7, i32 0, i32 0, i32 0), i32 0, i32 1 seq_cst seq_cst, align 4
  %23 = extractvalue { i32, i1 } %22, 1
  br i1 %23, label %24, label %26

24:                                               ; preds = %19
  %25 = tail call i64 (i64, ...) @syscall(i64 noundef 202, i64 noundef ptrtoint (%"class.(anonymous namespace)::Futex"* getelementptr inbounds ([4 x %"struct.(anonymous namespace)::Worker"], [4 x %"struct.(anonymous namespace)::Worker"]* @_ZN12_GLOBAL__N_17workersE, i64 0, i64 3, i32 7) to i64), i32 noundef 1, i32 noundef 1, i8* null, i8* null, i32 noundef 0) #5
  br label %26

26:                                               ; preds = %24, %19
  %27 = load i32, i32* getelementptr inbounds ([4 x %"struct.(anonymous namespace)::Worker"], [4 x %"struct.(anonymous namespace)::Worker"]* @_ZN12_GLOBAL__N_17workersE, i64 0, i64 3, i32 0), align 16, !tbaa !16
  %28 = tail call i32 @waitpid(i32 noundef %27, i32* noundef null, i32 noundef 0)
  ret void
}

declare i32 @waitpid(i32 noundef, i32* noundef, i32 noundef) local_unnamed_addr #4

; Function Attrs: mustprogress uwtable
define dso_local void @parallelFor(i32 noundef %0, i32 noundef %1, void (i32, i32)* noundef %2) local_unnamed_addr #3 personality i32 (...)* @__gxx_personality_v0 {
  %4 = alloca %struct.timespec, align 8
  %5 = alloca i32, align 4
  %6 = alloca %struct.timespec, align 8
  %7 = icmp sgt i32 %1, %0
  br i1 %7, label %9, label %8

8:                                                ; preds = %3
  tail call void %2(i32 noundef %0, i32 noundef %1)
  br label %299

9:                                                ; preds = %3
  %10 = sub nsw i32 %1, %0
  %11 = icmp ult i32 %10, 16
  br i1 %11, label %12, label %13

12:                                               ; preds = %9
  tail call void %2(i32 noundef %0, i32 noundef %1)
  br label %299

13:                                               ; preds = %9
  %14 = load i32, i32* @_ZL9lookupPtr, align 4, !tbaa !22
  br label %15

15:                                               ; preds = %40, %13
  %16 = phi i32 [ 0, %13 ], [ %41, %40 ]
  %17 = phi i32 [ %14, %13 ], [ %42, %40 ]
  %18 = icmp eq i32 %17, 16
  br i1 %18, label %19, label %20

19:                                               ; preds = %15
  store i32 0, i32* @_ZL9lookupPtr, align 4, !tbaa !22
  br label %20

20:                                               ; preds = %19, %15
  %21 = phi i32 [ 0, %19 ], [ %17, %15 ]
  %22 = zext i32 %21 to i64
  %23 = getelementptr inbounds [16 x %struct.ParallelForEntry], [16 x %struct.ParallelForEntry]* @_ZL13parallelCache, i64 0, i64 %22
  %24 = getelementptr inbounds [16 x %struct.ParallelForEntry], [16 x %struct.ParallelForEntry]* @_ZL13parallelCache, i64 0, i64 %22, i32 2
  %25 = load i8, i8* %24, align 4, !tbaa !23, !range !26
  %26 = icmp eq i8 %25, 0
  br i1 %26, label %40, label %27

27:                                               ; preds = %20
  %28 = getelementptr inbounds %struct.ParallelForEntry, %struct.ParallelForEntry* %23, i64 0, i32 0
  %29 = load void (i32, i32)*, void (i32, i32)** %28, align 8, !tbaa !27
  %30 = icmp eq void (i32, i32)* %29, %2
  br i1 %30, label %31, label %40

31:                                               ; preds = %27
  %32 = getelementptr inbounds [16 x %struct.ParallelForEntry], [16 x %struct.ParallelForEntry]* @_ZL13parallelCache, i64 0, i64 %22, i32 1
  %33 = load i32, i32* %32, align 8, !tbaa !28
  %34 = icmp eq i32 %33, %10
  br i1 %34, label %35, label %40

35:                                               ; preds = %31
  %36 = zext i32 %21 to i64
  %37 = getelementptr inbounds [16 x %struct.ParallelForEntry], [16 x %struct.ParallelForEntry]* @_ZL13parallelCache, i64 0, i64 %36, i32 3
  %38 = load i32, i32* %37, align 8, !tbaa !29
  %39 = add i32 %38, 1
  store i32 %39, i32* %37, align 8, !tbaa !29
  br label %166

40:                                               ; preds = %31, %27, %20
  %41 = add nuw nsw i32 %16, 1
  %42 = add i32 %21, 1
  store i32 %42, i32* @_ZL9lookupPtr, align 4, !tbaa !22
  %43 = icmp eq i32 %41, 16
  br i1 %43, label %44, label %15, !llvm.loop !30

44:                                               ; preds = %40
  %45 = load i8, i8* getelementptr inbounds ([16 x %struct.ParallelForEntry], [16 x %struct.ParallelForEntry]* @_ZL13parallelCache, i64 0, i64 0, i32 2), align 4, !tbaa !23, !range !26
  %46 = icmp eq i8 %45, 0
  br i1 %46, label %158, label %47

47:                                               ; preds = %44
  %48 = load i8, i8* getelementptr inbounds ([16 x %struct.ParallelForEntry], [16 x %struct.ParallelForEntry]* @_ZL13parallelCache, i64 0, i64 1, i32 2), align 4, !tbaa !23, !range !26
  %49 = icmp eq i8 %48, 0
  br i1 %49, label %158, label %50

50:                                               ; preds = %47
  %51 = load i8, i8* getelementptr inbounds ([16 x %struct.ParallelForEntry], [16 x %struct.ParallelForEntry]* @_ZL13parallelCache, i64 0, i64 2, i32 2), align 4, !tbaa !23, !range !26
  %52 = icmp eq i8 %51, 0
  br i1 %52, label %158, label %53

53:                                               ; preds = %50
  %54 = load i8, i8* getelementptr inbounds ([16 x %struct.ParallelForEntry], [16 x %struct.ParallelForEntry]* @_ZL13parallelCache, i64 0, i64 3, i32 2), align 4, !tbaa !23, !range !26
  %55 = icmp eq i8 %54, 0
  br i1 %55, label %158, label %56

56:                                               ; preds = %53
  %57 = load i8, i8* getelementptr inbounds ([16 x %struct.ParallelForEntry], [16 x %struct.ParallelForEntry]* @_ZL13parallelCache, i64 0, i64 4, i32 2), align 4, !tbaa !23, !range !26
  %58 = icmp eq i8 %57, 0
  br i1 %58, label %158, label %59

59:                                               ; preds = %56
  %60 = load i8, i8* getelementptr inbounds ([16 x %struct.ParallelForEntry], [16 x %struct.ParallelForEntry]* @_ZL13parallelCache, i64 0, i64 5, i32 2), align 4, !tbaa !23, !range !26
  %61 = icmp eq i8 %60, 0
  br i1 %61, label %158, label %62

62:                                               ; preds = %59
  %63 = load i8, i8* getelementptr inbounds ([16 x %struct.ParallelForEntry], [16 x %struct.ParallelForEntry]* @_ZL13parallelCache, i64 0, i64 6, i32 2), align 4, !tbaa !23, !range !26
  %64 = icmp eq i8 %63, 0
  br i1 %64, label %158, label %65

65:                                               ; preds = %62
  %66 = load i8, i8* getelementptr inbounds ([16 x %struct.ParallelForEntry], [16 x %struct.ParallelForEntry]* @_ZL13parallelCache, i64 0, i64 7, i32 2), align 4, !tbaa !23, !range !26
  %67 = icmp eq i8 %66, 0
  br i1 %67, label %158, label %68

68:                                               ; preds = %65
  %69 = load i8, i8* getelementptr inbounds ([16 x %struct.ParallelForEntry], [16 x %struct.ParallelForEntry]* @_ZL13parallelCache, i64 0, i64 8, i32 2), align 4, !tbaa !23, !range !26
  %70 = icmp eq i8 %69, 0
  br i1 %70, label %158, label %71

71:                                               ; preds = %68
  %72 = load i8, i8* getelementptr inbounds ([16 x %struct.ParallelForEntry], [16 x %struct.ParallelForEntry]* @_ZL13parallelCache, i64 0, i64 9, i32 2), align 4, !tbaa !23, !range !26
  %73 = icmp eq i8 %72, 0
  br i1 %73, label %158, label %74

74:                                               ; preds = %71
  %75 = load i8, i8* getelementptr inbounds ([16 x %struct.ParallelForEntry], [16 x %struct.ParallelForEntry]* @_ZL13parallelCache, i64 0, i64 10, i32 2), align 4, !tbaa !23, !range !26
  %76 = icmp eq i8 %75, 0
  br i1 %76, label %158, label %77

77:                                               ; preds = %74
  %78 = load i8, i8* getelementptr inbounds ([16 x %struct.ParallelForEntry], [16 x %struct.ParallelForEntry]* @_ZL13parallelCache, i64 0, i64 11, i32 2), align 4, !tbaa !23, !range !26
  %79 = icmp eq i8 %78, 0
  br i1 %79, label %158, label %80

80:                                               ; preds = %77
  %81 = load i8, i8* getelementptr inbounds ([16 x %struct.ParallelForEntry], [16 x %struct.ParallelForEntry]* @_ZL13parallelCache, i64 0, i64 12, i32 2), align 4, !tbaa !23, !range !26
  %82 = icmp eq i8 %81, 0
  br i1 %82, label %158, label %83

83:                                               ; preds = %80
  %84 = load i8, i8* getelementptr inbounds ([16 x %struct.ParallelForEntry], [16 x %struct.ParallelForEntry]* @_ZL13parallelCache, i64 0, i64 13, i32 2), align 4, !tbaa !23, !range !26
  %85 = icmp eq i8 %84, 0
  br i1 %85, label %158, label %86

86:                                               ; preds = %83
  %87 = load i8, i8* getelementptr inbounds ([16 x %struct.ParallelForEntry], [16 x %struct.ParallelForEntry]* @_ZL13parallelCache, i64 0, i64 14, i32 2), align 4, !tbaa !23, !range !26
  %88 = icmp eq i8 %87, 0
  br i1 %88, label %158, label %89

89:                                               ; preds = %86
  %90 = load i8, i8* getelementptr inbounds ([16 x %struct.ParallelForEntry], [16 x %struct.ParallelForEntry]* @_ZL13parallelCache, i64 0, i64 15, i32 2), align 4, !tbaa !23, !range !26
  %91 = icmp eq i8 %90, 0
  br i1 %91, label %158, label %92

92:                                               ; preds = %89
  %93 = load i32, i32* getelementptr inbounds ([16 x %struct.ParallelForEntry], [16 x %struct.ParallelForEntry]* @_ZL13parallelCache, i64 0, i64 0, i32 3), align 16, !tbaa !29
  %94 = load i32, i32* getelementptr inbounds ([16 x %struct.ParallelForEntry], [16 x %struct.ParallelForEntry]* @_ZL13parallelCache, i64 0, i64 1, i32 3), align 8, !tbaa !29
  %95 = icmp ult i32 %94, %93
  %96 = select i1 %95, i32 %94, i32 %93
  %97 = zext i1 %95 to i32
  %98 = load i32, i32* getelementptr inbounds ([16 x %struct.ParallelForEntry], [16 x %struct.ParallelForEntry]* @_ZL13parallelCache, i64 0, i64 2, i32 3), align 16, !tbaa !29
  %99 = icmp ult i32 %98, %96
  %100 = select i1 %99, i32 %98, i32 %96
  %101 = select i1 %99, i32 2, i32 %97
  %102 = load i32, i32* getelementptr inbounds ([16 x %struct.ParallelForEntry], [16 x %struct.ParallelForEntry]* @_ZL13parallelCache, i64 0, i64 3, i32 3), align 8, !tbaa !29
  %103 = icmp ult i32 %102, %100
  %104 = select i1 %103, i32 %102, i32 %100
  %105 = select i1 %103, i32 3, i32 %101
  %106 = load i32, i32* getelementptr inbounds ([16 x %struct.ParallelForEntry], [16 x %struct.ParallelForEntry]* @_ZL13parallelCache, i64 0, i64 4, i32 3), align 16, !tbaa !29
  %107 = icmp ult i32 %106, %104
  %108 = select i1 %107, i32 %106, i32 %104
  %109 = select i1 %107, i32 4, i32 %105
  %110 = load i32, i32* getelementptr inbounds ([16 x %struct.ParallelForEntry], [16 x %struct.ParallelForEntry]* @_ZL13parallelCache, i64 0, i64 5, i32 3), align 8, !tbaa !29
  %111 = icmp ult i32 %110, %108
  %112 = select i1 %111, i32 %110, i32 %108
  %113 = select i1 %111, i32 5, i32 %109
  %114 = load i32, i32* getelementptr inbounds ([16 x %struct.ParallelForEntry], [16 x %struct.ParallelForEntry]* @_ZL13parallelCache, i64 0, i64 6, i32 3), align 16, !tbaa !29
  %115 = icmp ult i32 %114, %112
  %116 = select i1 %115, i32 %114, i32 %112
  %117 = select i1 %115, i32 6, i32 %113
  %118 = load i32, i32* getelementptr inbounds ([16 x %struct.ParallelForEntry], [16 x %struct.ParallelForEntry]* @_ZL13parallelCache, i64 0, i64 7, i32 3), align 8, !tbaa !29
  %119 = icmp ult i32 %118, %116
  %120 = select i1 %119, i32 %118, i32 %116
  %121 = select i1 %119, i32 7, i32 %117
  %122 = load i32, i32* getelementptr inbounds ([16 x %struct.ParallelForEntry], [16 x %struct.ParallelForEntry]* @_ZL13parallelCache, i64 0, i64 8, i32 3), align 16, !tbaa !29
  %123 = icmp ult i32 %122, %120
  %124 = select i1 %123, i32 %122, i32 %120
  %125 = select i1 %123, i32 8, i32 %121
  %126 = load i32, i32* getelementptr inbounds ([16 x %struct.ParallelForEntry], [16 x %struct.ParallelForEntry]* @_ZL13parallelCache, i64 0, i64 9, i32 3), align 8, !tbaa !29
  %127 = icmp ult i32 %126, %124
  %128 = select i1 %127, i32 %126, i32 %124
  %129 = select i1 %127, i32 9, i32 %125
  %130 = load i32, i32* getelementptr inbounds ([16 x %struct.ParallelForEntry], [16 x %struct.ParallelForEntry]* @_ZL13parallelCache, i64 0, i64 10, i32 3), align 16, !tbaa !29
  %131 = icmp ult i32 %130, %128
  %132 = select i1 %131, i32 %130, i32 %128
  %133 = select i1 %131, i32 10, i32 %129
  %134 = load i32, i32* getelementptr inbounds ([16 x %struct.ParallelForEntry], [16 x %struct.ParallelForEntry]* @_ZL13parallelCache, i64 0, i64 11, i32 3), align 8, !tbaa !29
  %135 = icmp ult i32 %134, %132
  %136 = select i1 %135, i32 %134, i32 %132
  %137 = select i1 %135, i32 11, i32 %133
  %138 = load i32, i32* getelementptr inbounds ([16 x %struct.ParallelForEntry], [16 x %struct.ParallelForEntry]* @_ZL13parallelCache, i64 0, i64 12, i32 3), align 16, !tbaa !29
  %139 = icmp ult i32 %138, %136
  %140 = select i1 %139, i32 %138, i32 %136
  %141 = select i1 %139, i32 12, i32 %137
  %142 = load i32, i32* getelementptr inbounds ([16 x %struct.ParallelForEntry], [16 x %struct.ParallelForEntry]* @_ZL13parallelCache, i64 0, i64 13, i32 3), align 8, !tbaa !29
  %143 = icmp ult i32 %142, %140
  %144 = select i1 %143, i32 %142, i32 %140
  %145 = select i1 %143, i32 13, i32 %141
  %146 = load i32, i32* getelementptr inbounds ([16 x %struct.ParallelForEntry], [16 x %struct.ParallelForEntry]* @_ZL13parallelCache, i64 0, i64 14, i32 3), align 16, !tbaa !29
  %147 = icmp ult i32 %146, %144
  %148 = select i1 %147, i32 %146, i32 %144
  %149 = select i1 %147, i32 14, i32 %145
  %150 = load i32, i32* getelementptr inbounds ([16 x %struct.ParallelForEntry], [16 x %struct.ParallelForEntry]* @_ZL13parallelCache, i64 0, i64 15, i32 3), align 8, !tbaa !29
  %151 = icmp ult i32 %150, %148
  %152 = select i1 %151, i32 15, i32 %149
  %153 = zext i32 %152 to i64
  %154 = getelementptr inbounds [16 x %struct.ParallelForEntry], [16 x %struct.ParallelForEntry]* @_ZL13parallelCache, i64 0, i64 %153
  %155 = getelementptr inbounds %struct.ParallelForEntry, %struct.ParallelForEntry* %154, i64 0, i32 0
  store void (i32, i32)* %2, void (i32, i32)** %155, align 8, !tbaa !27
  %156 = getelementptr inbounds [16 x %struct.ParallelForEntry], [16 x %struct.ParallelForEntry]* @_ZL13parallelCache, i64 0, i64 %153, i32 1
  store i32 %10, i32* %156, align 8, !tbaa !28
  %157 = getelementptr inbounds [16 x %struct.ParallelForEntry], [16 x %struct.ParallelForEntry]* @_ZL13parallelCache, i64 0, i64 %153, i32 3
  store i32 1, i32* %157, align 8, !tbaa !29
  store i32 %152, i32* @_ZL9lookupPtr, align 4, !tbaa !22
  br label %166

158:                                              ; preds = %89, %86, %83, %80, %77, %74, %71, %68, %65, %62, %59, %56, %53, %50, %47, %44
  %159 = phi i32 [ 0, %44 ], [ 1, %47 ], [ 2, %50 ], [ 3, %53 ], [ 4, %56 ], [ 5, %59 ], [ 6, %62 ], [ 7, %65 ], [ 8, %68 ], [ 9, %71 ], [ 10, %74 ], [ 11, %77 ], [ 12, %80 ], [ 13, %83 ], [ 14, %86 ], [ 15, %89 ]
  %160 = phi i8* [ getelementptr inbounds ([16 x %struct.ParallelForEntry], [16 x %struct.ParallelForEntry]* @_ZL13parallelCache, i64 0, i64 0, i32 2), %44 ], [ getelementptr inbounds ([16 x %struct.ParallelForEntry], [16 x %struct.ParallelForEntry]* @_ZL13parallelCache, i64 0, i64 1, i32 2), %47 ], [ getelementptr inbounds ([16 x %struct.ParallelForEntry], [16 x %struct.ParallelForEntry]* @_ZL13parallelCache, i64 0, i64 2, i32 2), %50 ], [ getelementptr inbounds ([16 x %struct.ParallelForEntry], [16 x %struct.ParallelForEntry]* @_ZL13parallelCache, i64 0, i64 3, i32 2), %53 ], [ getelementptr inbounds ([16 x %struct.ParallelForEntry], [16 x %struct.ParallelForEntry]* @_ZL13parallelCache, i64 0, i64 4, i32 2), %56 ], [ getelementptr inbounds ([16 x %struct.ParallelForEntry], [16 x %struct.ParallelForEntry]* @_ZL13parallelCache, i64 0, i64 5, i32 2), %59 ], [ getelementptr inbounds ([16 x %struct.ParallelForEntry], [16 x %struct.ParallelForEntry]* @_ZL13parallelCache, i64 0, i64 6, i32 2), %62 ], [ getelementptr inbounds ([16 x %struct.ParallelForEntry], [16 x %struct.ParallelForEntry]* @_ZL13parallelCache, i64 0, i64 7, i32 2), %65 ], [ getelementptr inbounds ([16 x %struct.ParallelForEntry], [16 x %struct.ParallelForEntry]* @_ZL13parallelCache, i64 0, i64 8, i32 2), %68 ], [ getelementptr inbounds ([16 x %struct.ParallelForEntry], [16 x %struct.ParallelForEntry]* @_ZL13parallelCache, i64 0, i64 9, i32 2), %71 ], [ getelementptr inbounds ([16 x %struct.ParallelForEntry], [16 x %struct.ParallelForEntry]* @_ZL13parallelCache, i64 0, i64 10, i32 2), %74 ], [ getelementptr inbounds ([16 x %struct.ParallelForEntry], [16 x %struct.ParallelForEntry]* @_ZL13parallelCache, i64 0, i64 11, i32 2), %77 ], [ getelementptr inbounds ([16 x %struct.ParallelForEntry], [16 x %struct.ParallelForEntry]* @_ZL13parallelCache, i64 0, i64 12, i32 2), %80 ], [ getelementptr inbounds ([16 x %struct.ParallelForEntry], [16 x %struct.ParallelForEntry]* @_ZL13parallelCache, i64 0, i64 13, i32 2), %83 ], [ getelementptr inbounds ([16 x %struct.ParallelForEntry], [16 x %struct.ParallelForEntry]* @_ZL13parallelCache, i64 0, i64 14, i32 2), %86 ], [ getelementptr inbounds ([16 x %struct.ParallelForEntry], [16 x %struct.ParallelForEntry]* @_ZL13parallelCache, i64 0, i64 15, i32 2), %89 ]
  %161 = zext i32 %159 to i64
  %162 = getelementptr inbounds [16 x %struct.ParallelForEntry], [16 x %struct.ParallelForEntry]* @_ZL13parallelCache, i64 0, i64 %161
  store i8 1, i8* %160, align 4, !tbaa !23
  %163 = getelementptr inbounds %struct.ParallelForEntry, %struct.ParallelForEntry* %162, i64 0, i32 0
  store void (i32, i32)* %2, void (i32, i32)** %163, align 8, !tbaa !27
  %164 = getelementptr inbounds [16 x %struct.ParallelForEntry], [16 x %struct.ParallelForEntry]* @_ZL13parallelCache, i64 0, i64 %161, i32 1
  store i32 %10, i32* %164, align 8, !tbaa !28
  %165 = getelementptr inbounds [16 x %struct.ParallelForEntry], [16 x %struct.ParallelForEntry]* @_ZL13parallelCache, i64 0, i64 %161, i32 3
  store i32 1, i32* %165, align 8, !tbaa !29
  store i32 %159, i32* @_ZL9lookupPtr, align 4, !tbaa !22
  br label %166

166:                                              ; preds = %158, %92, %35
  %167 = phi %struct.ParallelForEntry* [ %162, %158 ], [ %154, %92 ], [ %23, %35 ]
  %168 = getelementptr inbounds %struct.ParallelForEntry, %struct.ParallelForEntry* %167, i64 0, i32 3
  %169 = load i32, i32* %168, align 8, !tbaa !29
  %170 = icmp ult i32 %169, 100
  br i1 %170, label %211, label %171

171:                                              ; preds = %166
  %172 = icmp ult i32 %169, 160
  br i1 %172, label %190, label %173

173:                                              ; preds = %171
  %174 = getelementptr inbounds %struct.ParallelForEntry, %struct.ParallelForEntry* %167, i64 0, i32 5
  %175 = load i32, i32* %174, align 8, !tbaa !31
  %176 = icmp eq i32 %175, 0
  br i1 %176, label %177, label %211

177:                                              ; preds = %173
  %178 = getelementptr inbounds %struct.ParallelForEntry, %struct.ParallelForEntry* %167, i64 0, i32 4, i64 0
  %179 = load i64, i64* %178, align 8, !tbaa !17
  %180 = getelementptr inbounds %struct.ParallelForEntry, %struct.ParallelForEntry* %167, i64 0, i32 4, i64 1
  %181 = load i64, i64* %180, align 8, !tbaa !17
  %182 = icmp slt i64 %181, %179
  %183 = zext i1 %182 to i32
  %184 = select i1 %182, i64 %181, i64 %179
  %185 = getelementptr inbounds %struct.ParallelForEntry, %struct.ParallelForEntry* %167, i64 0, i32 4, i64 2
  %186 = load i64, i64* %185, align 8, !tbaa !17
  %187 = icmp slt i64 %186, %184
  %188 = select i1 %187, i32 2, i32 %183
  store i32 %188, i32* %174, align 8, !tbaa !31
  %189 = icmp eq i32 %188, 0
  br i1 %189, label %210, label %205

190:                                              ; preds = %171
  %191 = trunc i32 %169 to i8
  %192 = add i8 %191, -100
  %193 = udiv i8 %192, 20
  %194 = zext i8 %193 to i32
  %195 = bitcast %struct.timespec* %6 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %195) #5
  %196 = call i32 @clock_gettime(i32 noundef 1, %struct.timespec* noundef nonnull %6) #5
  %197 = getelementptr inbounds %struct.timespec, %struct.timespec* %6, i64 0, i32 0
  %198 = load i64, i64* %197, align 8, !tbaa !32
  %199 = mul i64 %198, -1000000000
  %200 = getelementptr inbounds %struct.timespec, %struct.timespec* %6, i64 0, i32 1
  %201 = load i64, i64* %200, align 8, !tbaa !34
  %202 = sub i64 %199, %201
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %195) #5
  %203 = icmp ult i8 %192, 20
  br i1 %203, label %204, label %205

204:                                              ; preds = %190
  call void %2(i32 noundef %0, i32 noundef %1)
  br label %283

205:                                              ; preds = %190, %177
  %206 = phi i64 [ %202, %190 ], [ undef, %177 ]
  %207 = phi i32 [ %194, %190 ], [ %188, %177 ]
  %208 = xor i1 %172, true
  fence seq_cst
  %209 = bitcast i32* %5 to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* nonnull %209) #5
  store i32 0, i32* %5, align 4
  br label %215

210:                                              ; preds = %177
  call void %2(i32 noundef %0, i32 noundef %1)
  br label %299

211:                                              ; preds = %166, %173
  %212 = phi i32 [ %175, %173 ], [ 2, %166 ]
  fence seq_cst
  %213 = bitcast i32* %5 to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* nonnull %213) #5
  store i32 0, i32* %5, align 4
  %214 = icmp eq i32 %212, 31
  br i1 %214, label %228, label %215

215:                                              ; preds = %205, %211
  %216 = phi i1 [ %208, %205 ], [ true, %211 ]
  %217 = phi i32 [ %207, %205 ], [ %212, %211 ]
  %218 = phi i64 [ %206, %205 ], [ undef, %211 ]
  %219 = shl nuw i32 1, %217
  %220 = bitcast i32* %5 to %"struct.std::array"*
  %221 = lshr i32 %10, %217
  %222 = add nuw i32 %221, 3
  %223 = and i32 %222, -4
  %224 = add nsw i32 %219, -1
  %225 = zext i32 %224 to i64
  %226 = zext i32 %219 to i64
  %227 = ptrtoint void (i32, i32)* %2 to i64
  br label %235

228:                                              ; preds = %260, %211
  %229 = phi i1 [ true, %211 ], [ %216, %260 ]
  %230 = phi i32 [ 31, %211 ], [ %217, %260 ]
  %231 = phi i64 [ undef, %211 ], [ %218, %260 ]
  %232 = phi i64 [ 2147483648, %211 ], [ %226, %260 ]
  %233 = bitcast i32* %5 to %"struct.std::array"*
  %234 = bitcast i32* %5 to i8*
  br label %263

235:                                              ; preds = %260, %215
  %236 = phi i64 [ 0, %215 ], [ %261, %260 ]
  %237 = trunc i64 %236 to i32
  %238 = mul nsw i32 %223, %237
  %239 = add nsw i32 %238, %0
  %240 = add nsw i32 %239, %223
  %241 = icmp sgt i32 %240, %1
  %242 = icmp eq i64 %236, %225
  %243 = select i1 %242, i1 true, i1 %241
  %244 = select i1 %243, i32 %1, i32 %240
  %245 = icmp slt i32 %239, %244
  br i1 %245, label %246, label %260

246:                                              ; preds = %235
  %247 = getelementptr inbounds [4 x %"struct.(anonymous namespace)::Worker"], [4 x %"struct.(anonymous namespace)::Worker"]* @_ZN12_GLOBAL__N_17workersE, i64 0, i64 %236, i32 4
  %248 = bitcast %"struct.std::atomic.0"* %247 to i64*
  store atomic i64 %227, i64* %248 seq_cst, align 8
  %249 = getelementptr inbounds [4 x %"struct.(anonymous namespace)::Worker"], [4 x %"struct.(anonymous namespace)::Worker"]* @_ZN12_GLOBAL__N_17workersE, i64 0, i64 %236, i32 5, i32 0, i32 0
  store atomic i32 %239, i32* %249 seq_cst, align 16
  %250 = getelementptr inbounds [4 x %"struct.(anonymous namespace)::Worker"], [4 x %"struct.(anonymous namespace)::Worker"]* @_ZN12_GLOBAL__N_17workersE, i64 0, i64 %236, i32 6, i32 0, i32 0
  store atomic i32 %244, i32* %250 seq_cst, align 4
  %251 = getelementptr inbounds [4 x %"struct.(anonymous namespace)::Worker"], [4 x %"struct.(anonymous namespace)::Worker"]* @_ZN12_GLOBAL__N_17workersE, i64 0, i64 %236, i32 7
  %252 = getelementptr inbounds %"class.(anonymous namespace)::Futex", %"class.(anonymous namespace)::Futex"* %251, i64 0, i32 0, i32 0, i32 0
  %253 = cmpxchg i32* %252, i32 0, i32 1 seq_cst seq_cst, align 4
  %254 = extractvalue { i32, i1 } %253, 1
  br i1 %254, label %255, label %258

255:                                              ; preds = %246
  %256 = ptrtoint %"class.(anonymous namespace)::Futex"* %251 to i64
  %257 = call i64 (i64, ...) @syscall(i64 noundef 202, i64 noundef %256, i32 noundef 1, i32 noundef 1, i8* null, i8* null, i32 noundef 0) #5
  br label %258

258:                                              ; preds = %255, %246
  %259 = getelementptr inbounds %"struct.std::array", %"struct.std::array"* %220, i64 0, i32 0, i64 %236
  store i8 1, i8* %259, align 1, !tbaa !35
  br label %260

260:                                              ; preds = %258, %235
  %261 = add nuw nsw i64 %236, 1
  %262 = icmp eq i64 %261, %226
  br i1 %262, label %228, label %235, !llvm.loop !36

263:                                              ; preds = %279, %228
  %264 = phi i64 [ 0, %228 ], [ %280, %279 ]
  %265 = getelementptr inbounds %"struct.std::array", %"struct.std::array"* %233, i64 0, i32 0, i64 %264
  %266 = load i8, i8* %265, align 1, !tbaa !35, !range !26
  %267 = icmp eq i8 %266, 0
  br i1 %267, label %279, label %268

268:                                              ; preds = %263
  %269 = getelementptr inbounds [4 x %"struct.(anonymous namespace)::Worker"], [4 x %"struct.(anonymous namespace)::Worker"]* @_ZN12_GLOBAL__N_17workersE, i64 0, i64 %264, i32 8
  %270 = getelementptr inbounds %"class.(anonymous namespace)::Futex", %"class.(anonymous namespace)::Futex"* %269, i64 0, i32 0, i32 0, i32 0
  %271 = cmpxchg i32* %270, i32 1, i32 0 seq_cst seq_cst, align 4
  %272 = extractvalue { i32, i1 } %271, 1
  br i1 %272, label %279, label %273

273:                                              ; preds = %268
  %274 = ptrtoint %"class.(anonymous namespace)::Futex"* %269 to i64
  br label %275

275:                                              ; preds = %275, %273
  %276 = call i64 (i64, ...) @syscall(i64 noundef 202, i64 noundef %274, i32 noundef 0, i32 noundef 0, i8* null, i8* null, i32 noundef 0) #5
  %277 = cmpxchg i32* %270, i32 1, i32 0 seq_cst seq_cst, align 4
  %278 = extractvalue { i32, i1 } %277, 1
  br i1 %278, label %279, label %275, !llvm.loop !19

279:                                              ; preds = %275, %268, %263
  %280 = add nuw nsw i64 %264, 1
  %281 = icmp eq i64 %280, %232
  br i1 %281, label %282, label %263, !llvm.loop !37

282:                                              ; preds = %279
  fence seq_cst
  call void @llvm.lifetime.end.p0i8(i64 4, i8* nonnull %234) #5
  br i1 %229, label %299, label %283

283:                                              ; preds = %204, %282
  %284 = phi i32 [ %230, %282 ], [ 0, %204 ]
  %285 = phi i64 [ %231, %282 ], [ %202, %204 ]
  %286 = bitcast %struct.timespec* %4 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %286) #5
  %287 = call i32 @clock_gettime(i32 noundef 1, %struct.timespec* noundef nonnull %4) #5
  %288 = getelementptr inbounds %struct.timespec, %struct.timespec* %4, i64 0, i32 0
  %289 = load i64, i64* %288, align 8, !tbaa !32
  %290 = mul nsw i64 %289, 1000000000
  %291 = getelementptr inbounds %struct.timespec, %struct.timespec* %4, i64 0, i32 1
  %292 = load i64, i64* %291, align 8, !tbaa !34
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %286) #5
  %293 = zext i32 %284 to i64
  %294 = getelementptr inbounds %struct.ParallelForEntry, %struct.ParallelForEntry* %167, i64 0, i32 4, i64 %293
  %295 = load i64, i64* %294, align 8, !tbaa !17
  %296 = add i64 %292, %285
  %297 = add i64 %296, %290
  %298 = add i64 %297, %295
  store i64 %298, i64* %294, align 8, !tbaa !17
  br label %299

299:                                              ; preds = %210, %12, %283, %282, %8
  ret void
}

; Function Attrs: nounwind
declare i64 @syscall(i64 noundef, ...) local_unnamed_addr #2

; Function Attrs: nounwind
declare i32 @sched_setaffinity(i32 noundef, i64 noundef, %struct.cpu_set_t* noundef) local_unnamed_addr #2

declare i32 @__gxx_personality_v0(...)

; Function Attrs: nounwind
declare i32 @clock_gettime(i32 noundef, %struct.timespec* noundef) local_unnamed_addr #2

attributes #0 = { mustprogress nounwind uwtable "approx-func-fp-math"="true" "denormal-fp-math"="preserve-sign,preserve-sign" "denormal-fp-math-f32"="ieee,ieee" "frame-pointer"="none" "min-legal-vector-width"="0" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "unsafe-fp-math"="true" }
attributes #1 = { argmemonly mustprogress nofree nosync nounwind willreturn }
attributes #2 = { nounwind "approx-func-fp-math"="true" "denormal-fp-math"="preserve-sign,preserve-sign" "denormal-fp-math-f32"="ieee,ieee" "frame-pointer"="none" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "unsafe-fp-math"="true" }
attributes #3 = { mustprogress uwtable "approx-func-fp-math"="true" "denormal-fp-math"="preserve-sign,preserve-sign" "denormal-fp-math-f32"="ieee,ieee" "frame-pointer"="none" "min-legal-vector-width"="0" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "unsafe-fp-math"="true" }
attributes #4 = { "approx-func-fp-math"="true" "denormal-fp-math"="preserve-sign,preserve-sign" "denormal-fp-math-f32"="ieee,ieee" "frame-pointer"="none" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "unsafe-fp-math"="true" }
attributes #5 = { nounwind }

!llvm.module.flags = !{!0, !1, !2, !3}
!llvm.ident = !{!4}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 1}
!4 = !{!"Debian clang version 14.0.6"}
!5 = !{!6, !10, i64 8}
!6 = !{!"_ZTSN12_GLOBAL__N_16WorkerE", !7, i64 0, !10, i64 8, !11, i64 16, !11, i64 20, !12, i64 24, !14, i64 32, !14, i64 36, !15, i64 40, !15, i64 44}
!7 = !{!"int", !8, i64 0}
!8 = !{!"omnipotent char", !9, i64 0}
!9 = !{!"Simple C++ TBAA"}
!10 = !{!"any pointer", !8, i64 0}
!11 = !{!"_ZTSSt6atomicIjE"}
!12 = !{!"_ZTSSt6atomicIPFviiEE", !13, i64 0}
!13 = !{!"_ZTSSt13__atomic_baseIPFviiEE", !10, i64 0}
!14 = !{!"_ZTSSt6atomicIiE"}
!15 = !{!"_ZTSN12_GLOBAL__N_15FutexE", !11, i64 0}
!16 = !{!6, !7, i64 0}
!17 = !{!18, !18, i64 0}
!18 = !{!"long", !8, i64 0}
!19 = distinct !{!19, !20}
!20 = !{!"llvm.loop.mustprogress"}
!21 = distinct !{!21, !20}
!22 = !{!7, !7, i64 0}
!23 = !{!24, !25, i64 12}
!24 = !{!"_ZTS16ParallelForEntry", !10, i64 0, !7, i64 8, !25, i64 12, !7, i64 16, !8, i64 24, !7, i64 48}
!25 = !{!"bool", !8, i64 0}
!26 = !{i8 0, i8 2}
!27 = !{!24, !10, i64 0}
!28 = !{!24, !7, i64 8}
!29 = !{!24, !7, i64 16}
!30 = distinct !{!30, !20}
!31 = !{!24, !7, i64 48}
!32 = !{!33, !18, i64 0}
!33 = !{!"_ZTS8timespec", !18, i64 0, !18, i64 8}
!34 = !{!33, !18, i64 8}
!35 = !{!25, !25, i64 0}
!36 = distinct !{!36, !20}
!37 = distinct !{!37, !20}
