declare i32 @getint()

declare i32 @getch()

declare float @getfloat()

declare i32 @getarray(i32*)

declare i32 @getfarray(float*)

declare void @putint(i32)

declare void @putch(i32)

declare void @putfloat(float)

declare void @putarray(i32*)

declare void @putfarray(i32, float*)

declare void @putf()

declare void @starttime()

declare void @stoptime()

define i32 @main() {
bb0: ; entry
    ; nexts: bb2
    %0 = alloca i32
    store i32 0, i32* %0
    store i32 3, i32* %0
    br label %bb2

bb1:
    ; nexts: bb2
    br label %bb2

bb2: ; exit
    ; pres: bb0, bb1
    %1 = load i32, i32* %0
    ret i32 %1

}
In Function "main"
bb0 Prec: 
bb2 Prec: 	bb0

bb0 Succ: 	bb2
bb2 Succ: 

bb0 idom: null
bb2 idom: null

bb0 sdom: 	bb2
bb2 sdom: 	bb2

