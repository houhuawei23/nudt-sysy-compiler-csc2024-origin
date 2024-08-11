LoopParallel 过程主要步骤:

1. 分析出 Loop 信息
2. 提取出 LoopBody, 封装成 loop_body (new Function)
3. 构造 parallel_body(start, end) 函数
4. 通过 parallelFor(start, end, parallel_body) 调用并行库
5. parallelFor(start, end, parallel_body) 就是后端暴露给中端用于并行化的接口

```cpp
void parallel_body(int start, int end) {
    
}

int main() {
    call ParallelFor(beg, end, parallel_body);

}

```
after loop-extract:
```llvm
define i32 @main() {
bb0: ; entry
    ; nexts: bb2
    %0 = alloca [100 x i32] ; arr*
    br label %bb2 ; br next
bb1: ; exit
    ; pres: bb4
    ret i32 5
bb2: ; next
    ; pres: bb0
    ; nexts: bb3
    br label %bb3 ; br other
bb3: ; other
    ; pres: bb2
    ; nexts: bb5
    br label %bb5 ; br while1_judge
bb4: ; while1_next
    ; pres: bb5
    ; nexts: bb1
    br label %bb1 ; br exit
bb5: ; while1_judge
    ; pres: bb3, bb6
    ; nexts: bb7, bb4
    %1 = phi i32 [ 0, %bb3 ],[ %3, %bb6 ]
    %2 = icmp slt i32 %1, 100 
    br i1 %2, label %bb7, label %bb4
bb7:
    ; pres: bb5
    ; nexts: bb6
    call void @loop_body(i32 %1, [100 x i32]* %0)
    br label %bb6
bb6:
    ; pres: bb7
    ; nexts: bb5
    %3 = add i32 %1, 1
    br label %bb5 ; br while1_judge

}
define void @loop_body(i32 %0, [100 x i32]* %1) {
bb0:
    ; nexts: bb1
    br label %bb1 ; br while1_loop
bb1: ; while1_loop
    ; pres: bb0
    %2 = getelementptr [100 x i32], [100 x i32]* %1, i32 0, i32 %0
    store i32 %0, i32* %2
    ret void
}
```


```llvm
bb3: ; other
    ; pres: bb2
    ; nexts: bb5
    br label %bb5 ; br while1_judge
bb5: ; while1_judge
    ; pres: bb3, bb6
    ; nexts: bb7, bb4
    %1 = phi i32 [ 0, %bb3 ],[ %3, %bb6 ]
    %2 = icmp slt i32 %1, 100 
    br i1 %2, label %bb7, label %bb4
bb7:
    ; pres: bb5
    ; nexts: bb6
    call void @loop_body(i32 %1, [100 x i32]* %0)
    br label %bb6
bb6:
    ; pres: bb7
    ; nexts: bb5
    %3 = add i32 %1, 1
    br label %bb5 ; br while1_judge
bb4: ; while1_next
    ; pres: bb5
    ; nexts: bb1
    br label %bb1 ; br exit

```

```llvm
i32 main() {
bb3: ; other
    br label %bb_call
bb_call:
    call void @parallel_body(i32 0, i32 100)
    br label %bb4 ; br while1_next
bb4: ; while1_next
    br label %bb1 ; br exit
}

void parallel_body(i32 %beg, i32 %end) {
entry:
    br bb5
header: 
    %1 = phi i32 [ %beg, %entry ],[ %3, %latch ]
    %2 = icmp slt i32 %1, %end
    br i1 %2, label %body, label %exit
body:
    call void @loop_body(i32 %1, [100 x i32]* %0)
    br label %latch
latch:
    %3 = add i32 %1, 1
    br label %header
exit:
    ret void
}
```

```llvm
i32 main() {
bb3: ; other
    br label %bb_call
bb_call:
    %ptr = functionptr @parallel_body as i8*
    call parallelFor(i32 0, i32 100, i8* %ptr)
    ; call void @parallel_body(i32 0, i32 100)
    br label %bb4 ; br while1_next
bb4: ; while1_next
    br label %bb1 ; br exit
}

void parallel_body(i32 %beg, i32 %end) {
entry:
    br bb5
header: 
    %1 = phi i32 [ %beg, %entry ],[ %3, %latch ]
    %2 = icmp slt i32 %1, %end
    br i1 %2, label %body, label %exit
body:
    call void @loop_body(i32 %1, [100 x i32]* %0)
    br label %latch
latch:
    %3 = add i32 %1, 1
    br label %header
exit:
    ret void
}
```