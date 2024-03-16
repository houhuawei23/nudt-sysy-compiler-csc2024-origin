# SysYCompiler Project

创建于2024-2-17

本项目基本结构改编自: [sysy](https://gitee.com/xsu1989/sysy.git)

代码仓库: [SysYCompiler](https://gitee.com/triple-adventurer/sys-ycompiler.git)

开发人员(按拼音):侯华玮 简泽鑫 汤翔晟 杨俯众

## 项目基本结构和介绍

项目目前的文件结构：

- antlr/
- lib/
- src/
- test/
- antlr4compile.sh
- antlr4run.sh
- CMakeLists.txt
- README.md
- sysyc
  
antlr文件夹内部有antlr的.jar包和antlr-C++运行时库，antlr生成的各种文件的运行依赖这些运行时库，在`./antlr/antlr-runtime`中。

lib文件夹中存放动态库.so和静态库.a，他们是antlr-runtime的编译结果和src文件夹中各个SysY开头的cpp文件的编译结果，详细依赖关系见`./src/CMakeLists.txt`。

src文件夹中存放源代码，这包括了antlr生成的源代码和我们要开发的源代码。其中SysY开头的文件除了SysY.g4都是antlr生成的。

test文件夹与[sysy](https://gitee.com/xsu1989/sysy.git)中的test文件夹一致，存放sysY语言示例文件。

脚本antlr4compile.sh将使用antlr对`./src/SysY.g4`分析并生成相关代码文件，然后进行cmake和make命令。

脚本antlr4run.sh只调用cmake和make命令进行编译。

产生的可执行文件sysyc输出到当前文件夹。

## 安装

```shell
# 依赖
sudo apt-get update
sudo apt-get install -y build-essential uuid-dev libutfcpp-dev pkg-config make git cmake openjdk-11-jre

## 文档自动生成工具 doxygen
sudo apt-get install -y graphviz doxygen doxygen-gui
# gui doxygen
doxywizard
# config file 
doxygen Doxyfile

## antlr4 
sudo cp ./antlr/antlr-4.12.0-complete.jar /usr/local/lib/
# add to ~/.bashrc
export CLASSPATH=".:/usr/local/lib/antlr-4.12.0-complete.jar"
alias antlr4="java -jar /usr/local/lib/antlr-4.12.0-complete.jar"
alias grun="java org.antlr.v4.gui.TestRig"
```

## cmake

```shell
./cmake.sh

# runs
cmake -S . -B build
cmake --build build
```

## code到AST的分析

antlr4是一个编译器前端生成工具，可以生成多种目标语言的前端。本项目生成的目标语言是C++。

```shell
## antlr4cpp.sh
#!/bin/bash
main_path=$(pwd)

cd $main_path/src/antlr4

java -jar $main_path/antlr/antlr-4.12.0-complete.jar \
    -Dlanguage=Cpp -no-listener -visitor \
    SysY.g4 -o $main_path/src/.antlr4cpp

```

要运行antlr4，只需要用java解释器运行antlr.jar包即可，即`java -jar ../antlr/antlr-4.12.0-complete.jar`命令
其后紧跟的选项，都是antlr4工具的选项，可以在命令行中直接运行`java -jar ../antlr/antlr-4.12.0-complete.jar`命令查看帮助

这一步antlr产生的C++代码可以通过`-o`选项指定目录，或默认在当前目录`./src`

src中需要关注的文件有：

- ASTPrinter.cpp
- **ASTPrinter.h**
- CMakeLists.txt
- **SysY.g4**
- SysYBaseVisitor.cpp
- **SysYBaseVisitor.h**
- **sysyc.cpp**
- SysYParser.h
- SysYParser.cpp

## TO DO
- [] 环境配置 cmake 版本更新解决脚本
- 



## logs

- 2024.03.03:
  - 初步构建了 ir 和 visitor 所需的数据结构。
  - 暂时将 SysY.g4 更换成老师提供的版本。
  - `./cmake.sh`, 得到 main 可执行文件，可以测试 `./main ./test/main.sy`
  - 问题：visitor.hpp 无法 include ir 中的头文件，但是 visitCompUnit.cpp 可以引入？
  - 待完成：继续补充完善 ir 数据结构，重载 visitor 函数。
- 2024.03.12:
  - 整合了 ir 和 visitor 的数据结构
  - 生成 ir 的基本流程已打通，需要逐步细化
  - 实现了:
    - visit:
      - `visitFun.cpp`: vist func type, func
      - `visitDecl.cpp`: visit local decl scalar 
      - `visitStmt.cpp`: visit block, return stmt
      - `visitExp.cpp`: visit number exp
    - ir:
      - `builder.hpp`: create alloca, store, load, ret inst
      - `instructions.hpp`: AllocaInst, LoadInst, StoreInst, ReturnInst
        - but the print methods need to be modified 
      - `utils.hpp`: overload `<<` for convenience
      - nearly each class need to implement `print` method for print the readable ir
  - `test/steps/`: 逐步细化的测试用例，起初都是非常简单的用例，先跑起来！
  - **目前效果**：

```C
// test/steps/00_local.c
int main() {
    int l = 3;
    return 0;
}
```

```llvm
; test/steps/00_local_gen.ll
; main 生成的可读的 llvm ir 
; 可使用 > lli 00_local_gen.ll 运行, echo $? 查看返回值 (0)
define i32 @main() {
    %l = alloca i32
    store i32 3, i32* %l
    ret i32 0

}

; test/steps/00_local.ll
; clang -S -emit-llvm ./00_local.c 生成的 llvm ir (节选)
define dso_local i32 @main() #0 {
  %1 = alloca i32, align 4
  %2 = alloca i32, align 4
  store i32 0, i32* %1, align 4
  store i32 3, i32* %2, align 4
  ret i32 0
}
```

- 2024.3.14:
  1. create instructions empty class, add a lot method to implement
  2. modified the builder, add a lot help attributes and methods
  3. Constant gen method changed to template method
  4. delete IType enum,  add ValueId enum to Value, which specifies all subclasses of Value
  5. !!! add dyn_cast template function, very useful
  6. modified visitVarDef_beta, can deal with type cast and const
  7. modified visitNumberExp, can deal with float and int (hex/dec/oct)

- 2024.3.14:
- 1. change construc func in Constant, a constant's name is its value string
- 2. change name in Value, a variable's value is now in form:"%<number>"
- 3. realize visitVarExp (scalar)
- 4. realize create_load, LoadInst::print()
- 5. change SysY.g4, same with branch dev_frontend
- 6. realize IRBuilder::getvarname() (temp ver.)
- 7. change instruction.cpp:print(), use name() method to print variable name afterwards

**目前效果**
```C
// test/steps/00_local.c
int main() {
    int l = 3;
    float f=1.300;
    int g=l;
    float b=f;
    return b;
}
```

```llvm
; test/steps/00_local_gen.ll
; main 生成的可读的 llvm ir 
; 可使用 > lli 00_local_gen.ll 运行, echo $? 查看返回值 (0)
define i32 @main() {
    %1 = alloca i32
    store i32 5, i32* %1
    %2 = alloca float
    %3 = alloca i32
    %4 = load i32, i32* %1
    store i32 %4, i32* %3
    %5 = load i32, i32* %3
    ret i32 %5

}

; test/steps/00_local.ll
; clang -S -emit-llvm ./00_local.c 生成的 llvm ir (节选)
define dso_local i32 @main() #0 {
  %1 = alloca i32, align 4
  %2 = alloca i32, align 4
  %3 = alloca float, align 4
  %4 = alloca i32, align 4
  store i32 0, i32* %1, align 4
  store i32 5, i32* %2, align 4
  %5 = load i32, i32* %2, align 4
  store i32 %5, i32* %4, align 4
  %6 = load i32, i32* %4, align 4
  ret i32 %6
}
```
- 2024-3-15
  - 1.realize costant folding concerning addtion,substraction,multiplication,division and modulo
  - 2.realize visitParenExp
  - 3.realize ir::getMC(float f) in utils.hpp which is used to transform floating number to its machine code representaion

**当前效果**
```C
int main() {
    float b=2.0*3.0+4.0;
    int i=10%3;
    int c=10*(-9);
    int g=(2+3)*5.5;
    return c;
}
```

```llvm
; test/steps/00_local_gen.ll
; main 生成的可读的 llvm ir 
; 可使用 > lli 00_local_gen.ll 运行, echo $? 查看返回值 (0)
define i32 @main() {
    %1 = alloca float
    store float 0X4120000000000000, float* %1
    %2 = alloca i32
    store i32 1, i32* %2
    %3 = alloca i32
    store i32 -90, i32* %3
    %4 = alloca i32
    store i32 27, i32* %4
    %5 = load i32, i32* %3
    ret i32 %5

}

; test/steps/00_local.ll
; clang -S -emit-llvm ./00_local.c 生成的 llvm ir (节选)
define dso_local i32 @main() #0 {
  %1 = alloca i32, align 4
  %2 = alloca float, align 4
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  %5 = alloca i32, align 4
  store i32 0, i32* %1, align 4
  store float 1.000000e+01, float* %2, align 4
  store i32 1, i32* %3, align 4
  store i32 -90, i32* %4, align 4
  store i32 27, i32* %5, align 4
  %6 = load i32, i32* %4, align 4
  ret i32 %6
}
```
-2024-3-15
对于常量声明，如果右值直接可计算，不需要alloca和store，直接将该变量的name和值的地址插入符号表，支持声明时的类型转换 int->float float->int
int main()
{
    int a = 1;
    const int b = 2 + 1;
    return 0;
}
define i32 @main() {
    %1 = alloca i32
    store i32 1, i32* %1
    ret i32 0

}