@n = global i32 0

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

define i32 @swap(i32* %0, i32 %1, i32 %2) {
bb0: ; entry
    %3 = alloca i32* ; array*
    store i32* %0, i32** %3
    %4 = load i32*, i32** %3 ; load array
    %5 = getelementptr i32, i32* %4, i32 %1
    %6 = load i32, i32* %5
    %7 = load i32*, i32** %3 ; load array
    %8 = getelementptr i32, i32* %7, i32 %1
    %9 = load i32*, i32** %3 ; load array
    %10 = getelementptr i32, i32* %9, i32 %2
    %11 = load i32, i32* %10
    store i32 %11, i32* %8
    %12 = load i32*, i32** %3 ; load array
    %13 = getelementptr i32, i32* %12, i32 %2
    store i32 %6, i32* %13
    ret i32 0

}
define i32 @heap_ajust(i32* %0, i32 %1, i32 %2) {
bb0: ; entry
    ; nexts: bb2
    %3 = alloca i32* ; arr*
    store i32* %0, i32** %3
    %4 = mul i32 %1, 2
    %5 = add i32 %4, 1
    br label %bb2 ; br while1_judge

bb2: ; while1_judge
    ; pres: bb0, bb10
    ; nexts: bb3, bb14
    %6 = phi i32 [ %1, %bb0 ],[ %19, %bb10 ]
    %7 = phi i32 [ %5, %bb0 ],[ %30, %bb10 ]
    %8 = add i32 %2, 1
    %9 = icmp slt i32 %7, %8
    br i1 %9, label %bb3, label %bb14 ; br while1_loop, exit

bb3: ; while1_loop
    ; pres: bb2
    ; nexts: bb4, bb7
    %10 = icmp slt i32 %7, %2
    br i1 %10, label %bb4, label %bb7 ; br rhs_block, if1_merge

bb4: ; rhs_block
    ; pres: bb3
    ; nexts: bb5, bb7
    %11 = load i32*, i32** %3 ; load arr
    %12 = getelementptr i32, i32* %11, i32 %7
    %13 = load i32, i32* %12
    %14 = load i32*, i32** %3 ; load arr
    %15 = add i32 %7, 1
    %16 = getelementptr i32, i32* %14, i32 %15
    %17 = load i32, i32* %16
    %18 = icmp slt i32 %13, %17
    br i1 %18, label %bb5, label %bb7 ; br if1_then, if1_merge

bb5: ; if1_then
    ; pres: bb4
    ; nexts: bb7
    br label %bb7 ; br if1_merge

bb7: ; if1_merge
    ; pres: bb5, bb3, bb4
    ; nexts: bb10, bb14
