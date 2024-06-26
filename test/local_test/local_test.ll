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

define i32 @select_sort(i32* %0, i32 %1) {
bb0: ; entry
    ; nexts: bb1
    %2 = alloca i32
    %3 = alloca i32
    %4 = alloca i32* ; A*
    br label %bb1

bb1:
    ; pres: bb0
    ; nexts: bb2
    store i32* %0, i32** %4
    store i32 0, i32* %2
    br label %bb2 ; br while1_judge

bb2: ; while1_judge
    ; pres: bb1, bb12
    ; nexts: bb3, bb13
    %5 = load i32, i32* %2
    %6 = load i32, i32* %3
    %7 = sub i32 %1, 1
    %8 = icmp slt i32 %5, %7
    br i1 %8, label %bb3, label %bb13 ; br while1_loop, while1_next

bb3: ; while1_loop
    ; pres: bb2
    ; nexts: bb4
    %9 = add i32 %5, 1
    store i32 %9, i32* %2
    br label %bb4 ; br while2_judge

bb4: ; while2_judge
    ; pres: bb3, bb8
    ; nexts: bb5, bb9
    %10 = load i32, i32* %2
    %11 = load i32, i32* %2
    %12 = icmp slt i32 %10, %1
    br i1 %12, label %bb5, label %bb9 ; br while2_loop, while2_next

bb5: ; while2_loop
    ; pres: bb4
    ; nexts: bb6, bb7
    %13 = load i32*, i32** %4 ; load A
    %14 = getelementptr i32, i32* %13, i32 %11
    %15 = load i32, i32* %14
    %16 = load i32*, i32** %4 ; load A
    %17 = getelementptr i32, i32* %16, i32 %10
    %18 = load i32, i32* %17
    %19 = icmp sgt i32 %15, %18
    br i1 %19, label %bb6, label %bb7 ; br if1_then, if1_else

bb6: ; if1_then
    ; pres: bb5
    ; nexts: bb8
    br label %bb8 ; br if1_merge

bb7: ; if1_else
    ; pres: bb5
    ; nexts: bb8
    br label %bb8 ; br if1_merge

bb8: ; if1_merge
    ; pres: bb6, bb7
    ; nexts: bb4
    %20 = load i32, i32* %2
    %21 = add i32 %10, 1
    store i32 %21, i32* %2
    br label %bb4 ; br while2_judge

bb9: ; while2_next
    ; pres: bb4
    ; nexts: bb10, bb11
    %22 = icmp ne i32 %11, %5
    br i1 %22, label %bb10, label %bb11 ; br if2_then, if2_else

bb10: ; if2_then
    ; pres: bb9
    ; nexts: bb12
    %23 = load i32*, i32** %4 ; load A
    %24 = getelementptr i32, i32* %23, i32 %11
    %25 = load i32, i32* %24
    %26 = load i32*, i32** %4 ; load A
    %27 = getelementptr i32, i32* %26, i32 %11
    %28 = load i32*, i32** %4 ; load A
    %29 = getelementptr i32, i32* %28, i32 %5
    %30 = load i32, i32* %29
    store i32 %30, i32* %27
    %31 = load i32*, i32** %4 ; load A
    %32 = getelementptr i32, i32* %31, i32 %5
    store i32 %25, i32* %32
    store i32 %25, i32* %3
    br label %bb12 ; br if2_merge

bb11: ; if2_else
    ; pres: bb9
    ; nexts: bb12
    br label %bb12 ; br if2_merge

bb12: ; if2_merge
    ; pres: bb10, bb11
    ; nexts: bb2
    %33 = load i32, i32* %3
    %34 = add i32 %5, 1
    store i32 %34, i32* %2
    br label %bb2 ; br while1_judge

bb13: ; while1_next
    ; pres: bb2
    ; nexts: bb15
    br label %bb15 ; br exit

bb15: ; exit
    ; pres: bb13
    ret i32 0

}
define i32 @main() {
bb0: ; entry
    ; nexts: bb1
    %0 = alloca i32
    %1 = alloca [10 x i32] ; a*
    br label %bb1

bb1:
    ; pres: bb0
    ; nexts: bb2
    store i32 10, i32* @n
    %2 = getelementptr [10 x i32], [10 x i32]* %1, i32 0, i32 0
    %3 = getelementptr [10 x i32], [10 x i32]* %1, i32 0, i32 0
    store i32 4, i32* %3
    %4 = getelementptr [10 x i32], [10 x i32]* %1, i32 0, i32 1
    store i32 3, i32* %4
    %5 = getelementptr [10 x i32], [10 x i32]* %1, i32 0, i32 2
    store i32 9, i32* %5
    %6 = getelementptr [10 x i32], [10 x i32]* %1, i32 0, i32 3
    store i32 2, i32* %6
    %7 = getelementptr [10 x i32], [10 x i32]* %1, i32 0, i32 4
    store i32 0, i32* %7
    %8 = getelementptr [10 x i32], [10 x i32]* %1, i32 0, i32 5
    store i32 1, i32* %8
    %9 = getelementptr [10 x i32], [10 x i32]* %1, i32 0, i32 6
    store i32 6, i32* %9
    %10 = getelementptr [10 x i32], [10 x i32]* %1, i32 0, i32 7
    store i32 5, i32* %10
    %11 = getelementptr [10 x i32], [10 x i32]* %1, i32 0, i32 8
    store i32 7, i32* %11
    %12 = getelementptr [10 x i32], [10 x i32]* %1, i32 0, i32 9
    store i32 8, i32* %12
    %13 = getelementptr [10 x i32], [10 x i32]* %1, i32 0, i32 0
    %14 = load i32, i32* @n
    %15 = call i32 @select_sort(i32* %13, i32 %14)
    store i32 %15, i32* %0
    br label %bb2 ; br while3_judge

bb2: ; while3_judge
    ; pres: bb1, bb3
    ; nexts: bb3, bb4
    %16 = load i32, i32* %0
    %17 = load i32, i32* @n
    %18 = icmp slt i32 %16, %17
    br i1 %18, label %bb3, label %bb4 ; br while3_loop, while3_next

bb3: ; while3_loop
    ; pres: bb2
    ; nexts: bb2
    %19 = getelementptr [10 x i32], [10 x i32]* %1, i32 0, i32 %16
    %20 = load i32, i32* %19
    call void @putint(i32 %20)
    call void @putch(i32 10)
    %21 = add i32 %16, 1
    store i32 %21, i32* %0
    br label %bb2 ; br while3_judge

bb4: ; while3_next
    ; pres: bb2
    ; nexts: bb6
    br label %bb6 ; br exit

bb6: ; exit
    ; pres: bb4
    ret i32 0

}
