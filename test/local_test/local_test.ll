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

define i32 @ifWhile() {
bb0: ; entry
    ; nexts: bb7, bb8
    %0 = icmp eq i32 0, 5 ; 0 eq 5
    br i1 %0, label %bb7, label %bb8

bb1: ; if1_merge
    ; pres: bb2, bb6
    %1 = phi i32 [ %2, %bb2 ],[ %8, %bb6 ]
    ret i32 %1

bb2: ; while1_next
    ; pres: bb4
    ; nexts: bb1
    %2 = add i32 %4, 25
    br label %bb1 ; br if1_merge

bb3: ; while1_loop
    ; pres: bb4
    ; nexts: bb4
    %3 = add i32 %4, 2
    br label %bb4 ; br while1_judge

bb4: ; while1_judge
    ; pres: bb3, bb7
    ; nexts: bb3, bb2
    %4 = phi i32 [ %3, %bb3 ],[ 3, %bb7 ]
    %5 = icmp eq i32 %4, 2
    br i1 %5, label %bb3, label %bb2 ; br while1_loop, while1_next

bb5: ; while2_loop
    ; pres: bb6
    ; nexts: bb6
    %6 = mul i32 %8, 2
    %7 = add i32 %9, 1
    br label %bb6 ; br while2_judge

bb6: ; while2_judge
    ; pres: bb5, bb8
    ; nexts: bb5, bb1
    %8 = phi i32 [ %6, %bb5 ],[ 3, %bb8 ]
    %9 = phi i32 [ %7, %bb5 ],[ 0, %bb8 ]
    %10 = icmp slt i32 %9, 5
    br i1 %10, label %bb5, label %bb1 ; br while2_loop, if1_merge

bb7:
    ; pres: bb0
    ; nexts: bb4
    br label %bb4 ; br while1_judge

bb8:
    ; pres: bb0
    ; nexts: bb6
    br label %bb6 ; br while2_judge

}
define i32 @main() {
bb0: ; entry
    %0 = call i32 @ifWhile()
    ret i32 %0

}
