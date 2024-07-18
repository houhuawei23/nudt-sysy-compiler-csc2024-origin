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
    br label %bb2 ; br next

bb1: ; exit
    ; pres: bb4
    ret i32 0

bb2: ; next
    ; pres: bb0
    ; nexts: bb3
    br label %bb3 ; br other

bb3: ; other
    ; pres: bb2
    ; nexts: bb5
    br label %bb5

bb4:
    ; pres: bb6
    ; nexts: bb1
    call void @putint(i32 %0)
    br label %bb1 ; br exit

bb5:
    ; pres: bb3
    ; nexts: bb7
    br label %bb7

bb6:
    ; pres: bb9, bb11
    ; nexts: bb4
    %0 = phi i32 [ %5, %bb11 ],[ 5, %bb9 ]
    br label %bb4

bb7:
    ; pres: bb5
    ; nexts: bb8
    br label %bb8

bb8:
    ; pres: bb7
    ; nexts: bb9, bb12
    %1 = icmp eq i32 5, 6 ; 5 eq 6
    br i1 %1, label %bb9, label %bb12

bb9:
    ; pres: bb8, bb12
    ; nexts: bb6
    br label %bb6

bb10:
    ; pres: bb12
    ; nexts: bb16, bb14
    %2 = icmp eq i32 10, 10 ; 10 eq 10
    br i1 %2, label %bb16, label %bb14

bb11:
    ; pres: bb15
    ; nexts: bb6
    br label %bb6

bb12:
    ; pres: bb8
    ; nexts: bb9, bb10
    %3 = icmp eq i32 10, 11 ; 10 eq 11
    br i1 %3, label %bb9, label %bb10

bb13:
    ; pres: bb16
    ; nexts: bb15
    br label %bb15

bb14:
    ; pres: bb10, bb16
    ; nexts: bb20, bb18
    %4 = icmp eq i32 10, 10 ; 10 eq 10
    br i1 %4, label %bb20, label %bb18

bb15:
    ; pres: bb13, bb19
    ; nexts: bb11
    %5 = phi i32 [ %9, %bb19 ],[ 25, %bb13 ]
    br label %bb11

bb16:
    ; pres: bb10
    ; nexts: bb13, bb14
    %6 = icmp eq i32 5, 1 ; 5 eq 1
    br i1 %6, label %bb13, label %bb14

bb17:
    ; pres: bb20
    ; nexts: bb19
    %7 = add i32 5, 15 ; 5 add 15
    br label %bb19

bb18:
    ; pres: bb14, bb20
    ; nexts: bb19
    %8 = sub i32 0, 5 ; 0 sub 5
    br label %bb19

bb19:
    ; pres: bb17, bb18
    ; nexts: bb15
    %9 = phi i32 [ %8, %bb18 ],[ %7, %bb17 ]
    br label %bb15

bb20:
    ; pres: bb14
    ; nexts: bb17, bb18
    %10 = icmp eq i32 5, -5 ; 5 eq -5
    br i1 %10, label %bb17, label %bb18

}

