define i32 @main() {
bb0:     ; block
    %0 = alloca i32
    %1 = alloca [4 x [2 x i32]]
    %2 = getelementptr [4 x [2 x i32]], [4 x [2 x i32]]* %1, i32 0, i32 0
    %3 = getelementptr [2 x i32], [2 x i32]* %2, i32 0, i32 0
    store i32 1, i32* %3
    %4 = getelementptr i32, i32* %3, i32 1
    store i32 2, i32* %4
    %5 = getelementptr i32, i32* %4, i32 1
    store i32 3, i32* %5
    %6 = getelementptr i32, i32* %5, i32 1
    store i32 4, i32* %6
    %7 = getelementptr i32, i32* %6, i32 3
    store i32 7, i32* %7
    %4 = alloca [4 x [2 x i32]]
    %5 = getelementptr [4 x [2 x i32]], [4 x [2 x i32]]* %4, i32 0, i32 0
    %6 = getelementptr [2 x i32], [2 x i32]* %5, i32 0, i32 0
    %7 = alloca [4 x [2 x i32]]
    %8 = getelementptr [4 x [2 x i32]], [4 x [2 x i32]]* %7, i32 0, i32 0
    %9 = getelementptr [2 x i32], [2 x i32]* %8, i32 0, i32 0
    store i32 1, i32* %9
    %14 = getelementptr i32, i32* %9, i32 1
    store i32 2, i32* %14
    %15 = getelementptr i32, i32* %14, i32 1
    store i32 3, i32* %15
    %16 = getelementptr i32, i32* %15, i32 1
    store i32 4, i32* %16
    %17 = getelementptr i32, i32* %16, i32 1
    store i32 5, i32* %17
    %18 = getelementptr i32, i32* %17, i32 1
    store i32 6, i32* %18
    %19 = getelementptr i32, i32* %18, i32 1
    store i32 7, i32* %19
    %20 = getelementptr i32, i32* %19, i32 1
    store i32 8, i32* %20
    %10 = getelementptr [4 x [2 x i32]], [4 x [2 x i32]]* %1, i32 0, i32 3
    %11 = getelementptr [2 x i32], [2 x i32]* %10, i32 0, i32 0
    %12 = load i32, i32* %11
    %13 = alloca [4 x [2 x i32]]
    %14 = getelementptr [4 x [2 x i32]], [4 x [2 x i32]]* %13, i32 0, i32 0
    %15 = getelementptr [2 x i32], [2 x i32]* %14, i32 0, i32 0
    store i32 1, i32* %15
    %27 = getelementptr i32, i32* %15, i32 1
    store i32 2, i32* %27
    %28 = getelementptr i32, i32* %27, i32 1
    store i32 3, i32* %28
    %29 = getelementptr i32, i32* %28, i32 2
    store i32 5, i32* %29
    %30 = getelementptr i32, i32* %29, i32 1
    store i32 3, i32* %30
    %31 = getelementptr i32, i32* %30, i32 1
    store i32 %12, i32* %31
    %32 = getelementptr i32, i32* %31, i32 1
    store i32 8, i32* %32
    %16 = getelementptr [4 x [2 x i32]], [4 x [2 x i32]]* %13, i32 0, i32 2
    %17 = getelementptr [2 x i32], [2 x i32]* %16, i32 0, i32 1
    %18 = load i32, i32* %17
    %19 = getelementptr [4 x [2 x i32]], [4 x [2 x i32]]* %7, i32 0, i32 2
    %20 = getelementptr [2 x i32], [2 x i32]* %19, i32 0, i32 1
    %21 = load i32, i32* %20
    %22 = alloca [4 x [2 x [1 x i32]]]
    %23 = getelementptr [4 x [2 x [1 x i32]]], [4 x [2 x [1 x i32]]]* %22, i32 0, i32 0
    %24 = getelementptr [2 x [1 x i32]], [2 x [1 x i32]]* %23, i32 0, i32 0
    %25 = getelementptr [1 x i32], [1 x i32]* %24, i32 0, i32 0
    store i32 %18, i32* %25
    %43 = getelementptr i32, i32* %25, i32 1
    store i32 %21, i32* %43
    %44 = getelementptr i32, i32* %43, i32 1
    store i32 3, i32* %44
    %45 = getelementptr i32, i32* %44, i32 1
    store i32 4, i32* %45
    %46 = getelementptr i32, i32* %45, i32 1
    store i32 5, i32* %46
    %47 = getelementptr i32, i32* %46, i32 1
    store i32 6, i32* %47
    %48 = getelementptr i32, i32* %47, i32 1
    store i32 7, i32* %48
    %49 = getelementptr i32, i32* %48, i32 1
    store i32 8, i32* %49
    %26 = getelementptr [4 x [2 x [1 x i32]]], [4 x [2 x [1 x i32]]]* %22, i32 0, i32 3
    %27 = getelementptr [2 x [1 x i32]], [2 x [1 x i32]]* %26, i32 0, i32 1
    %28 = getelementptr [1 x i32], [1 x i32]* %27, i32 0, i32 0
    %29 = load i32, i32* %28
    %30 = getelementptr [4 x [2 x [1 x i32]]], [4 x [2 x [1 x i32]]]* %22, i32 0, i32 0
    %31 = getelementptr [2 x [1 x i32]], [2 x [1 x i32]]* %30, i32 0, i32 0
    %32 = getelementptr [1 x i32], [1 x i32]* %31, i32 0, i32 0
    %33 = load i32, i32* %32
    %34 = add i32 %29, %33
    %35 = getelementptr [4 x [2 x [1 x i32]]], [4 x [2 x [1 x i32]]]* %22, i32 0, i32 0
    %36 = getelementptr [2 x [1 x i32]], [2 x [1 x i32]]* %35, i32 0, i32 1
    %37 = getelementptr [1 x i32], [1 x i32]* %36, i32 0, i32 0
    %38 = load i32, i32* %37
    %39 = add i32 %34, %38
    %40 = getelementptr [4 x [2 x i32]], [4 x [2 x i32]]* %13, i32 0, i32 3
    %41 = getelementptr [2 x i32], [2 x i32]* %40, i32 0, i32 0
    %42 = load i32, i32* %41
    %43 = add i32 %39, %42
    store i32 %43, i32* %0
    br label %bb2

bb1:     ; block
    br label %bb2

bb2:     ; block
    %44 = load i32, i32* %0
    ret i32 %44

}
