@a = global [4 x [2 x [2 x [2 x i32]]]] [[2 x [2 x [2 x i32]]] [[2 x [2 x i32]] [[2 x i32] [i32 1, i32 2], [2 x i32] [i32 3, i32 4]], [2 x [2 x i32]] [[2 x i32] [i32 6, i32 7], [2 x i32] [i32 8, i32 9]]], [2 x [2 x [2 x i32]]] [[2 x [2 x i32]] [[2 x i32] [i32 10, i32 11], [2 x i32] [i32 12, i32 13]], [2 x [2 x i32]] [[2 x i32] [i32 14, i32 15], [2 x i32] [i32 16, i32 0]]], [2 x [2 x [2 x i32]]] [[2 x [2 x i32]] [[2 x i32] [i32 0, i32 0], [2 x i32] [i32 0, i32 0]], [2 x [2 x i32]] [[2 x i32] [i32 0, i32 0], [2 x i32] [i32 0, i32 0]]], [2 x [2 x [2 x i32]]] [[2 x [2 x i32]] [[2 x i32] [i32 0, i32 0], [2 x i32] [i32 0, i32 0]], [2 x [2 x i32]] [[2 x i32] [i32 0, i32 0], [2 x i32] [i32 0, i32 0]]]]

define i32 @main() {
bb0:     ; block
    %0 = alloca i32
    %1 = getelementptr [4 x [2 x [2 x [2 x i32]]]], [4 x [2 x [2 x [2 x i32]]]]* @a, i32 0, i32 1
    %2 = getelementptr [2 x [2 x [2 x i32]]], [2 x [2 x [2 x i32]]]* %1, i32 0, i32 1
    %3 = getelementptr [2 x [2 x i32]], [2 x [2 x i32]]* %2, i32 0, i32 1
    %4 = getelementptr [2 x i32], [2 x i32]* %3, i32 0, i32 1
    %5 = load i32, i32* %4
    store i32 %5, i32* %0
    br label %bb2

bb1:     ; block
    br label %bb2

bb2:     ; block
    %6 = load i32, i32* %0
    ret i32 %6

}
