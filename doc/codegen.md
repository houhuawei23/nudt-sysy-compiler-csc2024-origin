Global Value:

- has init: .data
- not init: .bss
- const: .rodata

Function:

- register allocation
- 为该function使用的每一个global value建立local label
- 检查是否有子函数调用
- 完成函数entry/头部工作？
- 遍历bb
- 完成尾部 exit?

StackTable符号表：记录每个local value在栈上的位置，每次都load store

Function 头部工作：
- 在.text段生成该Function的FunctionLabel以及.type等汇编伪指令
- 进入该Function后，保存上一级Function的FP
- 通过上一级Function的SP设置该Function的FP
- 更新该Function的SP (即开辟栈空间)

开辟函数栈空间
- sub sp, sp, #n
- 多大空间，需要计算

Function 尾部
- 恢复上一级Function的FP, SP
- 返回上一级Function

为所有basicblock生成汇编
- Label
- ir inst -> mir inst -> asm inst
- one by one translation

初始化局部变量和常数
- 常数在可表示的立即数范围内
- 超出范围
- 3 在text段构建常量池放置常数，使用时加载
  - `LDR <locallabel>`
  - 在LocalLabel处存放了常量，`<LocalLabel>` 需在ldr的寻址范围内（4KB）

## nanke ir->mir

```c++
for (all global in irmodule.globals){
    if (is function) {
        if (not has body): 
        else: 
    }
    else {
        if (has init value) {
            // .data
            // MIRDataStorage:: Storage data init
        }
        else {
            // .bss
            // MIRZeroStorage
        } 
    }

}
        
```