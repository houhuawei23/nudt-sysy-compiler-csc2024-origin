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