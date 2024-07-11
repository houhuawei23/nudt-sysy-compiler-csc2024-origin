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

define i32 @main() {
bb0: ; entry
    ; nexts: bb2
    %0 = alloca i32 ; retval*
    %1 = alloca [2 x i32] ; a*
    %2 = alloca i32 ; i*
    %3 = alloca i32 ; n*
    %4 = alloca i32 ; tmp*
    br label %bb2 ; br next

bb1: ; exit
    ; pres: bb4
    %5 = load i32, i32* %0 ; load retval
    ret i32 %5

bb2: ; next
    ; pres: bb0
    ; nexts: bb3
    store i32 0, i32* %0
    br label %bb3 ; br other

bb3: ; other
    ; pres: bb2
    ; nexts: bb6
    %6 = bitcast [2 x i32]* %1 to i8*
    call void @llvm.memset.p0i8.i64(i8* %6, i8 0, i64 8, i1 false)
    %7 = getelementptr [2 x i32], [2 x i32]* %1, i32 0, i32 0
    %8 = getelementptr i32, i32* %7, i32 0
    store i32 1, i32* %8
    %9 = getelementptr i32, i32* %8, i32 1
    store i32 2, i32* %9
    store i32 0, i32* %2
    store i32 2, i32* %3
    %10 = load i32, i32* %2 ; load i
    call void @putint(i32 %10)
    br label %bb6 ; br while1_judge

bb4: ; while1_next
    ; pres: bb6
    ; nexts: bb1
    store i32 0, i32* %0
    br label %bb1 ; br exit

bb5: ; while1_loop
    ; pres: bb6
    ; nexts: bb6
    %11 = load i32, i32* %2 ; load i
    %12 = getelementptr [2 x i32], [2 x i32]* %1, i32 0, i32 %11
    %13 = load i32, i32* %12
    store i32 %13, i32* %4
    %14 = load i32, i32* %2 ; load i
    %15 = add i32 %14, 1 ; i add 1
    store i32 %15, i32* %2
    br label %bb6 ; br while1_judge

bb6: ; while1_judge
    ; pres: bb3, bb5
    ; nexts: bb5, bb4
    %16 = load i32, i32* %2 ; load i
    %17 = load i32, i32* %3 ; load n
    %18 = icmp slt i32 %16, %17 ; i slt n
    br i1 %18, label %bb5, label %bb4 ; br while1_loop, while1_next

}