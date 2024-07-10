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

define i32 @swap(i32* %0, i32 %1, i32 %2) {
bb0: ; entry
    ; nexts: bb2
    %3 = alloca i32* ; array*
    br label %bb2 ; br next

bb1: ; exit
    ; pres: bb3
    ret i32 0

bb2: ; next
    ; pres: bb0
    ; nexts: bb3
    store i32* %0, i32** %3
    br label %bb3 ; br other

bb3: ; other
    ; pres: bb2
    ; nexts: bb1
    %4 = load i32*, i32** %3 ; load array
    %5 = getelementptr i32, i32* %4, i32 %1
    %6 = load i32, i32* %5
    %7 = load i32*, i32** %3 ; load array
    %8 = getelementptr i32, i32* %7, i32 %1
    %9 = load i32*, i32** %3 ; load array
    %10 = getelementptr i32, i32* %9, i32 %2
    %11 = load i32, i32* %10
    store i32 %11, i32* %8
    %12 = load i32*, i32** %3 ; load array
    %13 = getelementptr i32, i32* %12, i32 %2
    store i32 %6, i32* %13
    br label %bb1 ; br exit

}
define i32 @heap_ajust(i32* %0, i32 %1, i32 %2) {
bb0: ; entry
    ; nexts: bb2
    %3 = alloca i32* ; arr*
    br label %bb2 ; br next

bb1: ; exit
    ; pres: bb11, bb4
    ret i32 0

bb2: ; next
    ; pres: bb0
    ; nexts: bb3
    store i32* %0, i32** %3
    br label %bb3 ; br other

bb3: ; other
    ; pres: bb2
    ; nexts: bb6
    %4 = mul i32 %1, 2
    %5 = add i32 %4, 1
    br label %bb6 ; br while1_judge

bb4: ; while1_next
    ; pres: bb6
    ; nexts: bb1
    br label %bb1 ; br exit

bb5: ; while1_loop
    ; pres: bb6
    ; nexts: bb10, bb8
    %6 = icmp slt i32 %7, %2
    br i1 %6, label %bb10, label %bb8 ; br rhs_block, if1_else

bb6: ; while1_judge
    ; pres: bb3, bb13
    ; nexts: bb5, bb4
    %7 = phi i32 [ %5, %bb3 ],[ %31, %bb13 ]
    %8 = phi i32 [ %1, %bb3 ],[ %12, %bb13 ]
    %9 = add i32 %2, 1
    %10 = icmp slt i32 %7, %9
    br i1 %10, label %bb5, label %bb4 ; br while1_loop, while1_next

bb7: ; if1_then
    ; pres: bb10
    ; nexts: bb9
    %11 = add i32 %7, 1
    br label %bb9 ; br if1_merge

bb8: ; if1_else
    ; pres: bb5, bb10
    ; nexts: bb9
    br label %bb9 ; br if1_merge

bb9: ; if1_merge
    ; pres: bb7, bb8
    ; nexts: bb11, bb12
    %12 = phi i32 [ %7, %bb8 ],[ %11, %bb7 ]
    %13 = load i32*, i32** %3 ; load arr
    %14 = getelementptr i32, i32* %13, i32 %8
    %15 = load i32, i32* %14
    %16 = load i32*, i32** %3 ; load arr
    %17 = getelementptr i32, i32* %16, i32 %12
    %18 = load i32, i32* %17
    %19 = icmp sgt i32 %15, %18
    br i1 %19, label %bb11, label %bb12 ; br if2_then, if2_else

bb10: ; rhs_block
    ; pres: bb5
    ; nexts: bb7, bb8
    %20 = load i32*, i32** %3 ; load arr
    %21 = getelementptr i32, i32* %20, i32 %7
    %22 = load i32, i32* %21
    %23 = load i32*, i32** %3 ; load arr
    %24 = add i32 %7, 1
    %25 = getelementptr i32, i32* %23, i32 %24
    %26 = load i32, i32* %25
    %27 = icmp slt i32 %22, %26
    br i1 %27, label %bb7, label %bb8 ; br if1_then, if1_else

bb11: ; if2_then
    ; pres: bb9
    ; nexts: bb1
    br label %bb1 ; br exit

bb12: ; if2_else
    ; pres: bb9
    ; nexts: bb13
    %28 = load i32*, i32** %3 ; load arr
    %29 = call i32 @swap(i32* %28, i32 %8, i32 %12)
    %30 = mul i32 %12, 2
    %31 = add i32 %30, 1
    br label %bb13 ; br if2_merge

bb13: ; if2_merge
    ; pres: bb12
    ; nexts: bb6
    br label %bb6 ; br while1_judge

}
define i32 @heap_sort(i32* %0, i32 %1) {
bb0: ; entry
    ; nexts: bb2
    %2 = alloca i32* ; arr*
    br label %bb2 ; br next

bb1: ; exit
    ; pres: bb7
    ret i32 0

bb2: ; next
    ; pres: bb0
    ; nexts: bb3
    store i32* %0, i32** %2
    br label %bb3 ; br other

bb3: ; other
    ; pres: bb2
    ; nexts: bb6
    %3 = sdiv i32 %1, 2
    %4 = sub i32 %3, 1
    br label %bb6 ; br while1_judge

bb4: ; while1_next
    ; pres: bb6
    ; nexts: bb9
    %5 = sub i32 %1, 1
    br label %bb9 ; br while2_judge

bb5: ; while1_loop
    ; pres: bb6
    ; nexts: bb6
    %6 = sub i32 %1, 1
    %7 = load i32*, i32** %2 ; load arr
    %8 = call i32 @heap_ajust(i32* %7, i32 %10, i32 %6)
    %9 = sub i32 %10, 1
    br label %bb6 ; br while1_judge

bb6: ; while1_judge
    ; pres: bb3, bb5
    ; nexts: bb5, bb4
    %10 = phi i32 [ %4, %bb3 ],[ %9, %bb5 ]
    %11 = icmp sgt i32 %10, -1
    br i1 %11, label %bb5, label %bb4 ; br while1_loop, while1_next

bb7: ; while2_next
    ; pres: bb9
    ; nexts: bb1
    br label %bb1 ; br exit

bb8: ; while2_loop
    ; pres: bb9
    ; nexts: bb9
    %12 = load i32*, i32** %2 ; load arr
    %13 = call i32 @swap(i32* %12, i32 0, i32 %18)
    %14 = sub i32 %18, 1
    %15 = load i32*, i32** %2 ; load arr
    %16 = call i32 @heap_ajust(i32* %15, i32 0, i32 %14)
    %17 = sub i32 %18, 1
    br label %bb9 ; br while2_judge

bb9: ; while2_judge
    ; pres: bb4, bb8
    ; nexts: bb8, bb7
    %18 = phi i32 [ %5, %bb4 ],[ %17, %bb8 ]
    %19 = icmp sgt i32 %18, 0
    br i1 %19, label %bb8, label %bb7 ; br while2_loop, while2_next

}
define i32 @main() {
bb0: ; entry
    ; nexts: bb2
    %0 = alloca [10 x i32] ; a*
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
    ; nexts: bb6
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
    %13 = load i32, i32* @n
    %14 = call i32 @heap_sort(i32* %12, i32 %13)
    br label %bb6 ; br while1_judge

bb4: ; while1_next
    ; pres: bb6
    ; nexts: bb1
    br label %bb1 ; br exit

bb5: ; while1_loop
    ; pres: bb6
    ; nexts: bb6
    %15 = getelementptr [10 x i32], [10 x i32]* %0, i32 0, i32 %18
    %16 = load i32, i32* %15
    call void @putint(i32 %16)
    call void @putch(i32 10)
    %17 = add i32 %18, 1
    br label %bb6 ; br while1_judge

bb6: ; while1_judge
    ; pres: bb3, bb5
    ; nexts: bb5, bb4
    %18 = phi i32 [ %14, %bb3 ],[ %17, %bb5 ]
    %19 = load i32, i32* @n
    %20 = icmp slt i32 %18, %19
    br i1 %20, label %bb5, label %bb4 ; br while1_loop, while1_next

}
