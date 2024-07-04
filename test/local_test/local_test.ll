@a = global i32 0

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
    ; nexts: bb1
    %0 = alloca i32
    %1 = alloca i32
    %2 = alloca i32 ; retval*
    %3 = alloca i32 ; b*
    br label %bb1

bb1:
    ; pres: bb0
    ; nexts: bb5
    store i32 0, i32* %2
    store i32 10, i32* @a
    %4 = load i32, i32* @a

bb2:
    ; nexts: bb3
    br label %bb3 ; br exit

bb3: ; exit
    ; pres: bb2, bb4
    %5 = load i32, i32* %2 ; load retval
    ret i32 %5

bb4:
    ; pres: bb8
    ; nexts: bb3
    store i32 %10, i32* %3
    %6 = load i32, i32* %3 ; load b
    store i32 %6, i32* %2
    br label %bb3 ; br exit

bb5:
    ; pres: bb1
    ; nexts: bb6
    br label %bb6

bb6:
    ; pres: bb5
    ; nexts: bb8
    store i32 0, i32* %1
    store i32 %4, i32* %0
    %7 = load i32, i32* %0
    %8 = sub i32 %7, 1
    store i32 %8, i32* %0
    %9 = load i32, i32* %0
    store i32 %9, i32* %1
    br label %bb8

bb7:
    ; nexts: bb8

bb8:
    ; pres: bb6, bb7
    ; nexts: bb4
    %10 = load i32, i32* %1
    br label %bb4

}
