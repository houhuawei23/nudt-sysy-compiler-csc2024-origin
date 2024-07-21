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

define i32 @func(i32 %0) {
bb0: ; entry
    ; nexts: bb3
    br label %bb3 ; br while1_judge

bb1: ; while1_next
    ; pres: bb3
    %1 = add i32 %4, 1
    ret i32 %1

bb2: ; while1_loop
    ; pres: bb3
    ; nexts: bb3
    %2 = add i32 %0, 2
    %3 = add i32 %5, 1
    br label %bb3 ; br while1_judge

bb3: ; while1_judge
    ; pres: bb2, bb0
    ; nexts: bb2, bb1
    %4 = phi i32 [ undef, %bb0 ],[ %2, %bb2 ]
    %5 = phi i32 [ 0, %bb0 ],[ %3, %bb2 ]
    %6 = icmp slt i32 %5, 100
    br i1 %6, label %bb2, label %bb1 ; br while1_loop, while1_next

}
define i32 @main() {
bb0: ; entry
    %0 = call i32 @func(i32 3)
    ret i32 0

}
