define i32 @main() {
0:     ; block
    %1 = icmp ne i32 0, 2
    br i1 %1, label %5, label %2

2:     ; block
    %3 = alloca i32
    store i32 3, i32* %3
    %4 = load i32, i32* %3
    ret i32 %4
    br label %6

5:     ; block
    br label %6

6:     ; block
    ret i32 2

}
