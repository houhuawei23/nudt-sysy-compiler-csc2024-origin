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

define i32 @if_if_Else() {
bb0: ; entry
    ; nexts: bb1, bb5
    %0 = alloca i32
    store i32 0, i32* %0
    %1 = alloca i32
    store i32 5, i32* %1
    %2 = alloca i32
    store i32 10, i32* %2
    %3 = load i32, i32* %1
    %4 = icmp eq i32 %3, 5
    br i1 %4, label %bb1, label %bb5

bb1: ; if1_then
    ; pres: bb0
    ; nexts: bb2, bb3
    %5 = load i32, i32* %2
    %6 = icmp eq i32 %5, 10
    br i1 %6, label %bb2, label %bb3

bb2: ; if2_then
    ; pres: bb1
    ; nexts: bb4
    store i32 25, i32* %1
    br label %bb4

bb3: ; if2_else
    ; pres: bb1
    ; nexts: bb4
    br label %bb4

bb4: ; if2_merge
    ; pres: bb2, bb3
    ; nexts: bb6
    br label %bb6

bb5: ; if1_else
    ; pres: bb0
    ; nexts: bb6
    %7 = load i32, i32* %1
    %8 = add i32 %7, 15
    store i32 %8, i32* %1
    br label %bb6

bb6: ; if1_merge
    ; pres: bb4, bb5
    ; nexts: bb8
    %9 = load i32, i32* %1
    store i32 %9, i32* %0
    br label %bb8

bb7:
    ; nexts: bb8
    br label %bb8

bb8: ; exit
    ; pres: bb6, bb7
    %10 = load i32, i32* %0
    ret i32 %10

}
define i32 @main() {
bb0: ; entry
    ; nexts: bb2
    %0 = alloca i32
    store i32 0, i32* %0
    %1 = call i32 @if_if_Else()
    store i32 %1, i32* %0
    br label %bb2

bb1:
    ; nexts: bb2
    br label %bb2

bb2: ; exit
    ; pres: bb0, bb1
    %2 = load i32, i32* %0
    ret i32 %2

}
In Function "if_if_Else"
bb0 Prec: 
bb1 Prec: 	bb0
bb2 Prec: 	bb1
bb3 Prec: 	bb1
bb4 Prec: 	bb2	bb3
bb5 Prec: 	bb0
bb6 Prec: 	bb4	bb5
bb8 Prec: 	bb6

bb0 Succ: 	bb1	bb5
bb1 Succ: 	bb2	bb3
bb2 Succ: 	bb4
bb3 Succ: 	bb4
bb4 Succ: 	bb6
bb5 Succ: 	bb6
bb6 Succ: 	bb8
bb8 Succ: 

bb0 idom: null
bb1 idom: 	bb0
bb2 idom: 	bb1
bb3 idom: 	bb1
bb4 idom: 	bb1
bb5 idom: 	bb0
bb6 idom: 	bb0
bb8 idom: 	bb6

bb0 sdom: 	bb0
bb1 sdom: 	bb0
bb2 sdom: 	bb1
bb3 sdom: 	bb1
bb4 sdom: 	bb1
bb5 sdom: 	bb0
bb6 sdom: 	bb0
bb8 sdom: 	bb6

bb0 domTreeSons: bb1	bb5	bb6	
bb1 domTreeSons: bb2	bb3	bb4	
bb2 domTreeSons: 
bb3 domTreeSons: 
bb4 domTreeSons: 
bb5 domTreeSons: 
bb6 domTreeSons: bb8	
bb8 domTreeSons: 

bb0 domFrontier: 
bb1 domFrontier: bb6	
bb2 domFrontier: bb4	
bb3 domFrontier: bb4	
bb4 domFrontier: bb6	
bb5 domFrontier: bb6	
bb6 domFrontier: 
bb8 domFrontier: 
In Function "main"
bb0 Prec: 
bb2 Prec: 	bb0

bb0 Succ: 	bb2
bb2 Succ: 

bb0 idom: null
bb2 idom: null

bb0 sdom: null
bb2 sdom: null

bb0 domTreeSons: 
bb2 domTreeSons: 

bb0 domFrontier: 
bb2 domFrontier: 
