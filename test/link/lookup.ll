; ModuleID = './test/link/lookup.cpp'
source_filename = "./test/link/lookup.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

%"class.std::ios_base::Init" = type { i8 }
%struct.LUTEntry = type { i64, i32, i32 }

@_ZStL8__ioinit = internal global %"class.std::ios_base::Init" zeroinitializer, align 1
@__dso_handle = external hidden global i8
@llvm.global_ctors = appending global [1 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 65535, void ()* @_GLOBAL__sub_I_lookup.cpp, i8* null }]

; Function Attrs: noinline uwtable
define internal void @__cxx_global_var_init() #0 section ".text.startup" {
  call void @_ZNSt8ios_base4InitC1Ev(%"class.std::ios_base::Init"* noundef nonnull align 1 dereferenceable(1) @_ZStL8__ioinit)
  %1 = call i32 @__cxa_atexit(void (i8*)* bitcast (void (%"class.std::ios_base::Init"*)* @_ZNSt8ios_base4InitD1Ev to void (i8*)*), i8* getelementptr inbounds (%"class.std::ios_base::Init", %"class.std::ios_base::Init"* @_ZStL8__ioinit, i32 0, i32 0), i8* @__dso_handle) #3
  ret void
}

declare void @_ZNSt8ios_base4InitC1Ev(%"class.std::ios_base::Init"* noundef nonnull align 1 dereferenceable(1)) unnamed_addr #1

; Function Attrs: nounwind
declare void @_ZNSt8ios_base4InitD1Ev(%"class.std::ios_base::Init"* noundef nonnull align 1 dereferenceable(1)) unnamed_addr #2

; Function Attrs: nounwind
declare i32 @__cxa_atexit(void (i8*)*, i8*, i8*) #3

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define dso_local %struct.LUTEntry* @sysycCacheLookup(%struct.LUTEntry* noundef %0, i32 noundef %1, i32 noundef %2) #4 {
  %4 = alloca %struct.LUTEntry*, align 8
  %5 = alloca %struct.LUTEntry*, align 8
  %6 = alloca i32, align 4
  %7 = alloca i32, align 4
  %8 = alloca i64, align 8
  %9 = alloca i64, align 8
  %10 = alloca i64, align 8
  %11 = alloca i64, align 8
  %12 = alloca i32, align 4
  %13 = alloca i32, align 4
  %14 = alloca %struct.LUTEntry*, align 8
  %15 = alloca %struct.LUTEntry*, align 8
  store %struct.LUTEntry* %0, %struct.LUTEntry** %5, align 8
  store i32 %1, i32* %6, align 4
  store i32 %2, i32* %7, align 4
  %16 = load i32, i32* %6, align 4
  %17 = sext i32 %16 to i64
  %18 = shl i64 %17, 32
  %19 = load i32, i32* %7, align 4
  %20 = sext i32 %19 to i64
  %21 = or i64 %18, %20
  store i64 %21, i64* %8, align 8
  %22 = load i64, i64* %8, align 8
  %23 = urem i64 %22, 1021
  store i64 %23, i64* %9, align 8
  %24 = load i64, i64* %8, align 8
  %25 = urem i64 %24, 1019
  %26 = add i64 1, %25
  store i64 %26, i64* %10, align 8
  %27 = load i64, i64* %9, align 8
  store i64 %27, i64* %11, align 8
  store i32 5, i32* %12, align 4
  store i32 5, i32* %13, align 4
  br label %28

28:                                               ; preds = %3, %63
  %29 = load %struct.LUTEntry*, %struct.LUTEntry** %5, align 8
  %30 = load i64, i64* %11, align 8
  %31 = getelementptr inbounds %struct.LUTEntry, %struct.LUTEntry* %29, i64 %30
  store %struct.LUTEntry* %31, %struct.LUTEntry** %14, align 8
  %32 = load %struct.LUTEntry*, %struct.LUTEntry** %14, align 8
  %33 = getelementptr inbounds %struct.LUTEntry, %struct.LUTEntry* %32, i32 0, i32 2
  %34 = load i32, i32* %33, align 4
  %35 = icmp ne i32 %34, 0
  br i1 %35, label %41, label %36

36:                                               ; preds = %28
  %37 = load i64, i64* %8, align 8
  %38 = load %struct.LUTEntry*, %struct.LUTEntry** %14, align 8
  %39 = getelementptr inbounds %struct.LUTEntry, %struct.LUTEntry* %38, i32 0, i32 0
  store i64 %37, i64* %39, align 8
  %40 = load %struct.LUTEntry*, %struct.LUTEntry** %14, align 8
  store %struct.LUTEntry* %40, %struct.LUTEntry** %4, align 8
  br label %74

41:                                               ; preds = %28
  %42 = load %struct.LUTEntry*, %struct.LUTEntry** %14, align 8
  %43 = getelementptr inbounds %struct.LUTEntry, %struct.LUTEntry* %42, i32 0, i32 0
  %44 = load i64, i64* %43, align 8
  %45 = load i64, i64* %8, align 8
  %46 = icmp eq i64 %44, %45
  br i1 %46, label %47, label %49

47:                                               ; preds = %41
  %48 = load %struct.LUTEntry*, %struct.LUTEntry** %14, align 8
  store %struct.LUTEntry* %48, %struct.LUTEntry** %4, align 8
  br label %74

49:                                               ; preds = %41
  %50 = load i32, i32* %13, align 4
  %51 = add i32 %50, 1
  store i32 %51, i32* %13, align 4
  %52 = icmp uge i32 %51, 5
  br i1 %52, label %53, label %54

53:                                               ; preds = %49
  br label %64

54:                                               ; preds = %49
  %55 = load i64, i64* %10, align 8
  %56 = load i64, i64* %11, align 8
  %57 = add i64 %56, %55
  store i64 %57, i64* %11, align 8
  %58 = load i64, i64* %11, align 8
  %59 = icmp uge i64 %58, 1021
  br i1 %59, label %60, label %63

60:                                               ; preds = %54
  %61 = load i64, i64* %11, align 8
  %62 = sub i64 %61, 1021
  store i64 %62, i64* %11, align 8
  br label %63

63:                                               ; preds = %60, %54
  br label %28, !llvm.loop !6

64:                                               ; preds = %53
  %65 = load %struct.LUTEntry*, %struct.LUTEntry** %5, align 8
  %66 = load i64, i64* %9, align 8
  %67 = getelementptr inbounds %struct.LUTEntry, %struct.LUTEntry* %65, i64 %66
  store %struct.LUTEntry* %67, %struct.LUTEntry** %15, align 8
  %68 = load %struct.LUTEntry*, %struct.LUTEntry** %15, align 8
  %69 = getelementptr inbounds %struct.LUTEntry, %struct.LUTEntry* %68, i32 0, i32 2
  store i32 0, i32* %69, align 4
  %70 = load i64, i64* %8, align 8
  %71 = load %struct.LUTEntry*, %struct.LUTEntry** %15, align 8
  %72 = getelementptr inbounds %struct.LUTEntry, %struct.LUTEntry* %71, i32 0, i32 0
  store i64 %70, i64* %72, align 8
  %73 = load %struct.LUTEntry*, %struct.LUTEntry** %15, align 8
  store %struct.LUTEntry* %73, %struct.LUTEntry** %4, align 8
  br label %74

74:                                               ; preds = %64, %47, %36
  %75 = load %struct.LUTEntry*, %struct.LUTEntry** %4, align 8
  ret %struct.LUTEntry* %75
}

; Function Attrs: noinline uwtable
define internal void @_GLOBAL__sub_I_lookup.cpp() #0 section ".text.startup" {
  call void @__cxx_global_var_init()
  ret void
}

attributes #0 = { noinline uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #2 = { nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #3 = { nounwind }
attributes #4 = { mustprogress noinline nounwind optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }

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
