@a = constant i32 10
@b = constant i32 5
define i32 @main() {
bb0:     ; block
    %0 = alloca i32
    store i32 @b, i32* %0
    br label %bb2

bb1:     ; block
    br label %bb2

bb2:     ; block
    %1 = load i32, i32* %0
    ret i32 %1

}
