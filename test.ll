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

define i32 @fun(i32* %0, i32 %1) {
bb0: ; entry
    ; nexts: bb1
    %2 = alloca i32 ; retval*
    %3 = alloca i32* ; a*
    %4 = alloca i32 ; b*
    br label %bb1

bb1:
    ; pres: bb0
    ; nexts: bb3
    store i32 0, i32* %2
    store i32* %0, i32** %3
    store i32 %1, i32* %4
    store i32 0, i32* %2
    br label %bb3 ; br exit

bb2:
    ; nexts: bb3
    br label %bb3 ; br exit

bb3: ; exit
    ; pres: bb1, bb2
    %5 = load i32, i32* %2 ; load retval
    ret i32 %5

}
define i32 @main() {
bb0: ; entry
    ; nexts: bb1
    %0 = alloca i32 ; retval*
    %1 = alloca [2 x i32] ; a*
    %2 = alloca i32 ; b*
    br label %bb1

bb1:
    ; pres: bb0
    ; nexts: bb3
    store i32 0, i32* %0
    %3 = getelementptr [2 x i32], [2 x i32]* %1, i32 0, i32 0
    store i32 2, i32* %2
    %4 = getelementptr [2 x i32], [2 x i32]* %1, i32 0, i32 1
    store i32 23, i32* %4
    %5 = getelementptr [2 x i32], [2 x i32]* %1, i32 0, i32 0
    %6 = load i32, i32* %2 ; load b
    %7 = call i32 @fun(i32* %5, i32 %6)
    store i32 %7, i32* %0
    br label %bb3 ; br exit

bb2:
    ; nexts: bb3
    br label %bb3 ; br exit

bb3: ; exit
    ; pres: bb1, bb2
    %8 = load i32, i32* %0 ; load retval
    ret i32 %8

}
