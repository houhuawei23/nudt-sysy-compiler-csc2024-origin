@buf = global [2 x [100 x i32]] [[100 x i32] zeroinitializer, [100 x i32] zeroinitializer]

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

define void @merge_sort(i32 %0, i32 %1) {
bb0: ; entry
    ; nexts: bb1
    br label %bb1

bb1:
    ; pres: bb0
    ; nexts: bb2, bb4
    %2 = add i32 %0, 1
    %3 = icmp sge i32 %2, %1
    br i1 %3, label %bb2, label %bb4 ; br if1_then, if1_else

bb2: ; if1_then
    ; pres: bb1
    ret void

bb3:
    ; nexts: bb5
    br label %bb5 ; br if1_merge

bb4: ; if1_else
    ; pres: bb1
    ; nexts: bb5
    br label %bb5 ; br if1_merge

bb5: ; if1_merge
    ; pres: bb3, bb4
    ; nexts: bb6
    %4 = add i32 %0, %1
    %5 = sdiv i32 %4, 2
    call void @merge_sort(i32 %0, i32 %5)
    call void @merge_sort(i32 %5, i32 %1)
    br label %bb6 ; br while1_judge

bb6: ; while1_judge
    ; pres: bb5, bb11
    ; nexts: bb7, bb12
    %6 = icmp slt i32 %0, %5
    br i1 %6, label %bb7, label %bb12 ; br rhs_block, while1_next

bb7: ; rhs_block
    ; pres: bb6
    ; nexts: bb8, bb12
    %7 = icmp slt i32 %5, %1
    br i1 %7, label %bb8, label %bb12 ; br while1_loop, while1_next

bb8: ; while1_loop
    ; pres: bb7
    ; nexts: bb9, bb10
    %8 = getelementptr [2 x [100 x i32]], [2 x [100 x i32]]* @buf, i32 0, i32 0
    %9 = getelementptr [100 x i32], [100 x i32]* %8, i32 0, i32 %0
    %10 = load i32, i32* %9
    %11 = getelementptr [2 x [100 x i32]], [2 x [100 x i32]]* @buf, i32 0, i32 0
    %12 = getelementptr [100 x i32], [100 x i32]* %11, i32 0, i32 %5
    %13 = load i32, i32* %12
    %14 = icmp slt i32 %10, %13
    br i1 %14, label %bb9, label %bb10 ; br if2_then, if2_else

bb9: ; if2_then
    ; pres: bb8
    ; nexts: bb11
    %15 = getelementptr [2 x [100 x i32]], [2 x [100 x i32]]* @buf, i32 0, i32 1
    %16 = getelementptr [100 x i32], [100 x i32]* %15, i32 0, i32 %0
    %17 = getelementptr [2 x [100 x i32]], [2 x [100 x i32]]* @buf, i32 0, i32 0
    %18 = getelementptr [100 x i32], [100 x i32]* %17, i32 0, i32 %0
    %19 = load i32, i32* %18
    store i32 %19, i32* %16
    %20 = add i32 %0, 1
    br label %bb11 ; br if2_merge

bb10: ; if2_else
    ; pres: bb8
    ; nexts: bb11
    %21 = getelementptr [2 x [100 x i32]], [2 x [100 x i32]]* @buf, i32 0, i32 1
    %22 = getelementptr [100 x i32], [100 x i32]* %21, i32 0, i32 %0
    %23 = getelementptr [2 x [100 x i32]], [2 x [100 x i32]]* @buf, i32 0, i32 0
    %24 = getelementptr [100 x i32], [100 x i32]* %23, i32 0, i32 %5
    %25 = load i32, i32* %24
    store i32 %25, i32* %22
    %26 = add i32 %5, 1
    br label %bb11 ; br if2_merge

bb11: ; if2_merge
    ; pres: bb9, bb10
    ; nexts: bb6
    %27 = add i32 %0, 1
    br label %bb6 ; br while1_judge

bb12: ; while1_next
    ; pres: bb6, bb7
    ; nexts: bb13
    br label %bb13 ; br while2_judge

bb13: ; while2_judge
    ; pres: bb12, bb14
    ; nexts: bb14, bb15
    %28 = icmp slt i32 %0, %5
    br i1 %28, label %bb14, label %bb15 ; br while2_loop, while2_next

bb14: ; while2_loop
    ; pres: bb13
    ; nexts: bb13
    %29 = getelementptr [2 x [100 x i32]], [2 x [100 x i32]]* @buf, i32 0, i32 1
    %30 = getelementptr [100 x i32], [100 x i32]* %29, i32 0, i32 %0
    %31 = getelementptr [2 x [100 x i32]], [2 x [100 x i32]]* @buf, i32 0, i32 0
    %32 = getelementptr [100 x i32], [100 x i32]* %31, i32 0, i32 %0
    %33 = load i32, i32* %32
    store i32 %33, i32* %30
    %34 = add i32 %0, 1
    %35 = add i32 %0, 1
    br label %bb13 ; br while2_judge

bb15: ; while2_next
    ; pres: bb13
    ; nexts: bb16
    br label %bb16 ; br while3_judge

bb16: ; while3_judge
    ; pres: bb15, bb17
    ; nexts: bb17, bb18
    %36 = icmp slt i32 %5, %1
    br i1 %36, label %bb17, label %bb18 ; br while3_loop, while3_next

bb17: ; while3_loop
    ; pres: bb16
    ; nexts: bb16
    %37 = getelementptr [2 x [100 x i32]], [2 x [100 x i32]]* @buf, i32 0, i32 1
    %38 = getelementptr [100 x i32], [100 x i32]* %37, i32 0, i32 %0
    %39 = getelementptr [2 x [100 x i32]], [2 x [100 x i32]]* @buf, i32 0, i32 0
    %40 = getelementptr [100 x i32], [100 x i32]* %39, i32 0, i32 %5
    %41 = load i32, i32* %40
    store i32 %41, i32* %38
    %42 = add i32 %5, 1
    %43 = add i32 %0, 1
    br label %bb16 ; br while3_judge

bb18: ; while3_next
    ; pres: bb16
    ; nexts: bb19
    br label %bb19 ; br while4_judge

bb19: ; while4_judge
    ; pres: bb18, bb20
    ; nexts: bb20, bb21
    %44 = icmp slt i32 %0, %1
    br i1 %44, label %bb20, label %bb21 ; br while4_loop, while4_next

bb20: ; while4_loop
    ; pres: bb19
    ; nexts: bb19
    %45 = getelementptr [2 x [100 x i32]], [2 x [100 x i32]]* @buf, i32 0, i32 0
    %46 = getelementptr [100 x i32], [100 x i32]* %45, i32 0, i32 %0
    %47 = getelementptr [2 x [100 x i32]], [2 x [100 x i32]]* @buf, i32 0, i32 1
    %48 = getelementptr [100 x i32], [100 x i32]* %47, i32 0, i32 %0
    %49 = load i32, i32* %48
    store i32 %49, i32* %46
    %50 = add i32 %0, 1
    br label %bb19 ; br while4_judge

bb21: ; while4_next
    ; pres: bb19
    ; nexts: bb22
    br label %bb22 ; br exit

bb22: ; exit
    ; pres: bb21
    ret void

}
define i32 @main() {
bb0: ; entry
    ; nexts: bb1
    br label %bb1

bb1:
    ; pres: bb0
    ; nexts: bb3
    %0 = getelementptr [2 x [100 x i32]], [2 x [100 x i32]]* @buf, i32 0, i32 0
    %1 = getelementptr [100 x i32], [100 x i32]* %0, i32 0, i32 0
    %2 = call i32 @getarray(i32* %1)
    call void @merge_sort(i32 0, i32 %2)
    %3 = getelementptr [2 x [100 x i32]], [2 x [100 x i32]]* @buf, i32 0, i32 0
    %4 = getelementptr [100 x i32], [100 x i32]* %3, i32 0, i32 0
    call void @putarray(i32 %2, i32* %4)
    br label %bb3 ; br exit

bb2:
    ; nexts: bb3
    br label %bb3 ; br exit

bb3: ; exit
    ; pres: bb1, bb2
    ret i32 0

}
