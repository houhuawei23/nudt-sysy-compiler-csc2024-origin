name a enter the if stmt
name b0 enter the if stmt
name _c enter the if stmt
define i32 @main() {
bb0:     ; block
    %0 = alloca i32
    %1 = alloca i32
    %2 = alloca i32
    %3 = alloca i32
    store i32 1, i32* %1
    store i32 2, i32* %2
    store i32 3, i32* %3
    %4 = load i32, i32* %2
    %5 = load i32, i32* %3
    %6 = add i32 %4, %5
    store i32 %6, i32* %0
    br label %bb2

bb1:     ; block
    br label %bb2

bb2:     ; block
    %7 = load i32, i32* %0
    ret i32 %7

}
