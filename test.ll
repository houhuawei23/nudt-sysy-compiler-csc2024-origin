@a = global i32 3

@b = global i32 5

define i32 @main() {
0:     ; block
    %1 = alloca i32
    store i32 5, i32* %1
    %2 = load i32, i32* %1
    %3 = load i32, i32* @b
    %4 = add i32 %2, %3
    ret i32 %4

}
