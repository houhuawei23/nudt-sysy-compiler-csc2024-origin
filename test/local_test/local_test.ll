@n = global i32 0

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

define i32 @QuickSort(i32* %0, i32 %1, i32 %2) {
bb0: ; entry
    ; nexts: bb1, bb2
    %3 = icmp slt i32 %1, %2
    br i1 %3, label %bb1, label %bb2 ; br if1_then, if1_merge

bb1: ; if1_then
    ; pres: bb0
    ; nexts: bb4
    %4 = getelementptr i32, i32* %0, i32 %1
    %5 = load i32, i32* %4
    br label %bb4 ; br while1_judge

bb2: ; if1_merge
    ; pres: bb3, bb0
    %6 = phi i32 
    ret i32 0

bb3: ; while1_next
    ; pres: bb4
    ; nexts: bb2
    %7 = phi i32 [ %15, %bb4 ]
    %8 = getelementptr i32, i32* %0, i32 %15
    store i32 %5, i32* %8
    %9 = sub i32 %15, 1
    %10 = call i32 @QuickSort(i32* %0, i32 %1, i32 %9)
    %11 = add i32 %15, 1
    %12 = call i32 @QuickSort(i32* %0, i32 %11, i32 %2)
    br label %bb2 ; br if1_merge

bb4: ; while1_judge
    ; pres: bb1, bb16
    ; nexts: bb3, bb17
    %13 = phi i32 
    %14 = phi i32 [ %2, %bb1 ],[ %43, %bb16 ]
    %15 = phi i32 [ %1, %bb1 ],[ %34, %bb16 ]
    %16 = icmp slt i32 %15, %14
    br i1 %16, label %bb17, label %bb3

bb5: ; while2_next
    ; pres: bb18, bb19
    ; nexts: bb9, bb10
    %17 = phi i32 
    %18 = icmp slt i32 %15, %20
    br i1 %18, label %bb9, label %bb10 ; br if2_then, if2_merge

bb6: ; while2_loop
    ; pres: bb8
    ; nexts: bb7
    %19 = sub i32 %20, 1
    br label %bb7 ; br while2_judge

bb7: ; while2_judge
    ; pres: bb6, bb17
    ; nexts: bb8, bb18
    %20 = phi i32 [ %19, %bb6 ],[ %14, %bb17 ]
    %21 = icmp slt i32 %15, %20
    br i1 %21, label %bb8, label %bb18

bb8: ; rhs_block
    ; pres: bb7
    ; nexts: bb6, bb19
    %22 = getelementptr i32, i32* %0, i32 %20
    %23 = load i32, i32* %22
    %24 = sub i32 %5, 1
    %25 = icmp sgt i32 %23, %24
    br i1 %25, label %bb6, label %bb19

bb9: ; if2_then
    ; pres: bb5
    ; nexts: bb10
    %26 = getelementptr i32, i32* %0, i32 %15
    %27 = getelementptr i32, i32* %0, i32 %20
    %28 = load i32, i32* %27
    store i32 %28, i32* %26
    %29 = add i32 %15, 1
    br label %bb10 ; br if2_merge

bb10: ; if2_merge
    ; pres: bb9, bb5
    ; nexts: bb13
    %30 = phi i32 [ %29, %bb9 ],[ %15, %bb5 ]
    br label %bb13 ; br while3_judge

bb11: ; while3_next
    ; pres: bb13, bb14
    ; nexts: bb15, bb16
    %31 = phi i32 [ %34, %bb13 ],[ %34, %bb14 ]
    %32 = icmp slt i32 %34, %20
    br i1 %32, label %bb15, label %bb16 ; br if3_then, if3_merge

bb12: ; while3_loop
    ; pres: bb14
    ; nexts: bb13
    %33 = add i32 %34, 1
    br label %bb13 ; br while3_judge

bb13: ; while3_judge
    ; pres: bb10, bb12
    ; nexts: bb14, bb11
    %34 = phi i32 [ %30, %bb10 ],[ %33, %bb12 ]
    %35 = icmp slt i32 %34, %20
    br i1 %35, label %bb14, label %bb11 ; br rhs_block, while3_next

bb14: ; rhs_block
    ; pres: bb13
    ; nexts: bb12, bb11
    %36 = getelementptr i32, i32* %0, i32 %34
    %37 = load i32, i32* %36
    %38 = icmp slt i32 %37, %5
    br i1 %38, label %bb12, label %bb11 ; br while3_loop, while3_next

bb15: ; if3_then
    ; pres: bb11
    ; nexts: bb16
    %39 = getelementptr i32, i32* %0, i32 %20
    %40 = getelementptr i32, i32* %0, i32 %34
    %41 = load i32, i32* %40
    store i32 %41, i32* %39
    %42 = sub i32 %20, 1
    br label %bb16 ; br if3_merge

bb16: ; if3_merge
    ; pres: bb15, bb11
    ; nexts: bb4
    %43 = phi i32 [ %42, %bb15 ],[ %20, %bb11 ]
    br label %bb4 ; br while1_judge

bb17:
    ; pres: bb4
    ; nexts: bb7
    br label %bb7 ; br while2_judge

bb18:
    ; pres: bb7
    ; nexts: bb5
    %44 = phi i32 [ %20, %bb7 ]
    br label %bb5 ; br while2_next

bb19:
    ; pres: bb8
    ; nexts: bb5
    %45 = phi i32 [ %20, %bb8 ]
    br label %bb5 ; br while2_next

}
define i32 @main() {
bb0: ; entry
    ; nexts: bb3
    %0 = alloca [10 x i32] ; a*
    store i32 10, i32* @n
    %1 = getelementptr [10 x i32], [10 x i32]* %0, i32 0, i32 0
    %2 = getelementptr [10 x i32], [10 x i32]* %0, i32 0, i32 0
    store i32 4, i32* %2
    %3 = getelementptr [10 x i32], [10 x i32]* %0, i32 0, i32 1
    store i32 3, i32* %3
    %4 = getelementptr [10 x i32], [10 x i32]* %0, i32 0, i32 2
    store i32 9, i32* %4
    %5 = getelementptr [10 x i32], [10 x i32]* %0, i32 0, i32 3
    store i32 2, i32* %5
    %6 = getelementptr [10 x i32], [10 x i32]* %0, i32 0, i32 4
    store i32 0, i32* %6
    %7 = getelementptr [10 x i32], [10 x i32]* %0, i32 0, i32 5
    store i32 1, i32* %7
    %8 = getelementptr [10 x i32], [10 x i32]* %0, i32 0, i32 6
    store i32 6, i32* %8
    %9 = getelementptr [10 x i32], [10 x i32]* %0, i32 0, i32 7
    store i32 5, i32* %9
    %10 = getelementptr [10 x i32], [10 x i32]* %0, i32 0, i32 8
    store i32 7, i32* %10
    %11 = getelementptr [10 x i32], [10 x i32]* %0, i32 0, i32 9
    store i32 8, i32* %11
    %12 = getelementptr [10 x i32], [10 x i32]* %0, i32 0, i32 0
    %13 = call i32 @QuickSort(i32* %12, i32 0, i32 9)
    br label %bb3 ; br while1_judge

bb1: ; while1_next
    ; pres: bb3
    ret i32 0

bb2: ; while1_loop
    ; pres: bb3
    ; nexts: bb3
    %14 = getelementptr [10 x i32], [10 x i32]* %0, i32 0, i32 %17
    %15 = load i32, i32* %14
    call void @putint(i32 %15)
    call void @putch(i32 10)
    %16 = add i32 %17, 1
    br label %bb3 ; br while1_judge

bb3: ; while1_judge
    ; pres: bb2, bb0
    ; nexts: bb2, bb1
    %17 = phi i32 [ %13, %bb0 ],[ %16, %bb2 ]
    %18 = load i32, i32* @n
    %19 = icmp slt i32 %17, %18
    br i1 %19, label %bb2, label %bb1 ; br while1_loop, while1_next

}
