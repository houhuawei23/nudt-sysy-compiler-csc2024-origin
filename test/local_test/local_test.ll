@ascii_0 = constant i32 48

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

define i32 @my_getint() {
bb0: ; entry
    ; nexts: bb1
    %0 = alloca i32
    %1 = alloca i32
    %2 = alloca i32
    br label %bb1

bb1:
    ; pres: bb0
    ; nexts: bb2
    br label %bb2 ; br while1_judge

bb2: ; while1_judge
    ; pres: bb1, bb5
    ; nexts: bb3, bb0
    %3 = load i32, i32* %2
    %4 = icmp ne i32 1, 0 ; 1 ne 0
    br i1 %4, label %bb3, label %bb0

bb3: ; while1_loop
    ; pres: bb2
    ; nexts: bb5, bb4
    %5 = call i32 @getch()
    %6 = sub i32 %5, 48
    %7 = icmp slt i32 %6, 0
    br i1 %7, label %bb5, label %bb4 ; br if1_then, rhs_block

bb4: ; rhs_block
    ; pres: bb3
    ; nexts: bb5, bb7
    %8 = icmp sgt i32 %6, 9
    br i1 %8, label %bb5, label %bb7 ; br if1_then, if1_else

bb5: ; if1_then
    ; pres: bb3, bb4
    ; nexts: bb2
    store i32 %6, i32* %2
    br label %bb2 ; br while1_judge

bb7: ; if1_else
    ; pres: bb4
    ; nexts: bb10
    store i32 %6, i32* %1
    br label %bb10 ; br while1_next

bb10: ; while1_next
    ; pres: bb7, bb0
    ; nexts: bb11
    %9 = load i32, i32* %1
    store i32 %9, i32* %0
    br label %bb11 ; br while2_judge

bb11: ; while2_judge
    ; pres: bb10, bb17
    ; nexts: bb12, bb18
    %10 = load i32, i32* %0
    %11 = icmp ne i32 1, 0 ; 1 ne 0
    br i1 %11, label %bb12, label %bb18 ; br while2_loop, while2_next

bb12: ; while2_loop
    ; pres: bb11
    ; nexts: bb13, bb15
    %12 = call i32 @getch()
    %13 = sub i32 %12, 48
    %14 = icmp sge i32 %13, 0
    br i1 %14, label %bb13, label %bb15 ; br rhs_block, if2_else

bb13: ; rhs_block
    ; pres: bb12
    ; nexts: bb14, bb15
    %15 = icmp sle i32 %13, 9
    br i1 %15, label %bb14, label %bb15 ; br if2_then, if2_else

bb14: ; if2_then
    ; pres: bb13
    ; nexts: bb17
    %16 = mul i32 %10, 10
    %17 = add i32 %16, %13
    br label %bb17 ; br if2_merge

bb15: ; if2_else
    ; pres: bb12, bb13
    ; nexts: bb18
    br label %bb18 ; br while2_next

bb17: ; if2_merge
    ; pres: bb14
    ; nexts: bb11
    store i32 %17, i32* %0
    br label %bb11 ; br while2_judge

bb18: ; while2_next
    ; pres: bb11, bb15
    ; nexts: bb20
    br label %bb20 ; br exit

bb20: ; exit
    ; pres: bb18
    ret i32 %10

bb0:
    ; pres: bb2
    ; nexts: bb10
    store i32 %3, i32* %1
    br label %bb10 ; br while1_next

}
define void @my_putint(i32 %0) {
bb0: ; entry
    ; nexts: bb1
    %1 = alloca i32
    %2 = alloca i32
    %3 = alloca i32
    %4 = alloca [16 x i32] ; b*
    br label %bb1

bb1:
    ; pres: bb0
    ; nexts: bb2
    %5 = getelementptr [16 x i32], [16 x i32]* %4, i32 0, i32 0
    store i32 0, i32* %3
    store i32 %0, i32* %2
    br label %bb2 ; br while3_judge

bb2: ; while3_judge
    ; pres: bb1, bb3
    ; nexts: bb3, bb4
    %6 = load i32, i32* %2
    %7 = load i32, i32* %3
    %8 = icmp sgt i32 %6, 0
    br i1 %8, label %bb3, label %bb4 ; br while3_loop, while3_next

bb3: ; while3_loop
    ; pres: bb2
    ; nexts: bb2
    %9 = getelementptr [16 x i32], [16 x i32]* %4, i32 0, i32 %7
    %10 = srem i32 %6, 10
    %11 = add i32 %10, 48
    store i32 %11, i32* %9
    %12 = sdiv i32 %6, 10
    %13 = add i32 %7, 1
    store i32 %13, i32* %3
    store i32 %12, i32* %2
    br label %bb2 ; br while3_judge

bb4: ; while3_next
    ; pres: bb2
    ; nexts: bb5
    store i32 %7, i32* %1
    br label %bb5 ; br while4_judge

bb5: ; while4_judge
    ; pres: bb4, bb6
    ; nexts: bb6, bb7
    %14 = load i32, i32* %1
    %15 = icmp sgt i32 %14, 0
    br i1 %15, label %bb6, label %bb7 ; br while4_loop, while4_next

bb6: ; while4_loop
    ; pres: bb5
    ; nexts: bb5
    %16 = sub i32 %14, 1
    %17 = getelementptr [16 x i32], [16 x i32]* %4, i32 0, i32 %16
    %18 = load i32, i32* %17
    call void @putch(i32 %18)
    store i32 %16, i32* %1
    br label %bb5 ; br while4_judge

bb7: ; while4_next
    ; pres: bb5
    ; nexts: bb8
    br label %bb8 ; br exit

bb8: ; exit
    ; pres: bb7
    ret void

}
define i32 @main() {
bb0: ; entry
    ; nexts: bb1
    %0 = alloca i32
    br label %bb1

bb1:
    ; pres: bb0
    ; nexts: bb2
    %1 = call i32 @my_getint()
    store i32 %1, i32* %0
    br label %bb2 ; br while5_judge

bb2: ; while5_judge
    ; pres: bb1, bb3
    ; nexts: bb3, bb4
    %2 = load i32, i32* %0
    %3 = icmp sgt i32 %2, 0
    br i1 %3, label %bb3, label %bb4 ; br while5_loop, while5_next

bb3: ; while5_loop
    ; pres: bb2
    ; nexts: bb2
    %4 = call i32 @my_getint()
    call void @my_putint(i32 %4)
    call void @putch(i32 10)
    %5 = sub i32 %2, 1
    store i32 %5, i32* %0
    br label %bb2 ; br while5_judge

bb4: ; while5_next
    ; pres: bb2
    ; nexts: bb6
    br label %bb6 ; br exit

bb6: ; exit
    ; pres: bb4
    ret i32 0

}
