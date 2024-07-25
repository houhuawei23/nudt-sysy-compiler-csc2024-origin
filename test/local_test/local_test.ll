declare void @llvm.memset.p0i8.i64(i8* nocapture writeonly, i8, i64, i1 immarg)
declare i32 @getint()

declare i32 @getch()

declare float @getfloat()

declare i32 @getarray(i32*)

declare i32 @getfarray(float*)

declare void @putint(i32)

declare void @putch(i32)

declare void @putfloat(float)

declare void @putarray(i32, i32*)

declare void @putfarray(i32, float*)

declare void @putf()

declare void @starttime()

declare void @stoptime()

declare void @_sysy_starttime()

declare void @_sysy_stoptime()

define i32 @exgcd(i32 %0, i32 %1, i32* %2, i32* %3) {
bb0: ; entry
    ; nexts: bb2, bb3
    %4 = icmp eq i32 %1, 0
    br i1 %4, label %bb2, label %bb3 ; br if1_then, if1_else

bb1: ; exit
    ; pres: bb2, bb3
    %5 = phi i32 [ %9, %bb3 ],[ %0, %bb2 ]
    ret i32 %5

bb2: ; if1_then
    ; pres: bb0
    ; nexts: bb1
    %6 = getelementptr i32, i32* %2, i32 0
    store i32 1, i32* %6
    %7 = getelementptr i32, i32* %3, i32 0
    store i32 0, i32* %7
    br label %bb1 ; br exit

bb3: ; if1_else
    ; pres: bb0
    ; nexts: bb1
    %8 = srem i32 %0, %1
    %9 = call i32 @exgcd(i32 %1, i32 %8, i32* %2, i32* %3)
    %10 = getelementptr i32, i32* %2, i32 0
    %11 = load i32, i32* %10
    %12 = getelementptr i32, i32* %2, i32 0
    %13 = getelementptr i32, i32* %3, i32 0
    %14 = load i32, i32* %13
    store i32 %14, i32* %12
    %15 = getelementptr i32, i32* %3, i32 0
    %16 = sdiv i32 %0, %1
    %17 = getelementptr i32, i32* %3, i32 0
    %18 = load i32, i32* %17
    %19 = mul i32 %16, %18
    %20 = sub i32 %11, %19
    store i32 %20, i32* %15
    br label %bb1 ; br exit

}
define i32 @main() {
bb0: ; entry
    %0 = alloca [1 x i32] ; x*
    %1 = alloca [1 x i32] ; y*
    %2 = bitcast [1 x i32]* %0 to i8*
    call void @llvm.memset.p0i8.i64(i8* %2, i8 0, i64 4, i1 false)
    %3 = getelementptr [1 x i32], [1 x i32]* %0, i32 0, i32 0
    %4 = getelementptr i32, i32* %3, i32 0
    store i32 1, i32* %4
    %5 = bitcast [1 x i32]* %1 to i8*
    call void @llvm.memset.p0i8.i64(i8* %5, i8 0, i64 4, i1 false)
    %6 = getelementptr [1 x i32], [1 x i32]* %1, i32 0, i32 0
    %7 = getelementptr i32, i32* %6, i32 0
    store i32 1, i32* %7
    %8 = getelementptr [1 x i32], [1 x i32]* %0, i32 0, i32 0
    %9 = getelementptr [1 x i32], [1 x i32]* %1, i32 0, i32 0
    %10 = call i32 @exgcd(i32 7, i32 15, i32* %8, i32* %9)
    %11 = getelementptr [1 x i32], [1 x i32]* %0, i32 0, i32 0
    %12 = getelementptr [1 x i32], [1 x i32]* %0, i32 0, i32 0
    %13 = load i32, i32* %12
    %14 = srem i32 %13, 15
    %15 = add i32 %14, 15
    %16 = srem i32 %15, 15
    store i32 %16, i32* %11
    %17 = getelementptr [1 x i32], [1 x i32]* %0, i32 0, i32 0
    %18 = load i32, i32* %17
    call void @putint(i32 %18)
    ret i32 0

}
