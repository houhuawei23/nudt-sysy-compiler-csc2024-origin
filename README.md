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
## useful commands

```bash
# clang
## gen llvm ir
clang -S -emit-llvm test.c 
clang -S --target=riscv64 -emit-llvm test.c -o test.rv64.ll
clang -S --target=arm -emit-llvm test.c -o test.rv64.ll

## print registered targets
clang -print-targets

# arm        - ARM
# arm64      - ARM64 (little endian)
# arm64_32   - ARM64 (little endian ILP32)
# armeb      - ARM (big endian)
# riscv32    - 32-bit RISC-V
# riscv64    - 64-bit RISC-V
# x86        - 32-bit X86: Pentium-Pro and above
# x86-64     - 64-bit X86: EM64T and AMD64

-mtargetos=<value>      
# Set the deployment target to be the specified OS and OS version
--offload-arch=<value>  CUDA offloading device architecture (e.g. sm_35), or HIP offloading target ID in the form of a device architecture followed by target ID features delimited by a colon. Each target ID feature is a pre-defined string followed by a plus or minus sign (e.g. gfx908:xnack+:sramecc-).  May be specified more than once.
--offload=<value>       Specify comma-separated list of offloading target triples (CUDA and HIP only)
-print-effective-triple 
# Print the effective target triple
-print-multiarch        
# Print the multiarch target triple
-print-supported-cpus   
# Print supported cpu models for the given target (if target is not specified, it will print the supported cpus for the default target)
-print-target-triple    
# Print the normalized target triple
-print-targets          
# Print the registered targets
--target=<value>        
# Generate code for the given target

# lli: interprete llvm ir file
clang -S -emit-llvm test.c -o test.ll
lli test.ll

# llc: compile llvmir file to assembly code
llc test.ll -o test.s
llc -march=riscv64 test.ll -o test.rv64.s

./clang -S -emit-llvm --target=armv7 -mfloat-abi=hard test.c
./llc -march=arm -float-abi=hard test.ll
arm-linux-gnueabihf-gcc

clang -S --target=riscv64 -emit-llvm test.c -o test.rv64.ll # gen .ll
llc -march=riscv64 test.rv64.ll -o test.rv64.s
riscv64-linux-gnu-gcc -march=rv64gc t.s  
# qemu
qemu-riscv64 -L /usr/riscv64-linux-gnu/ a.out 

llc -march=riscv64 -mcpu=generic-rv64 test.rv64.ll -o test.rv64.s
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

## 链接静态库
1. 生成静态链接库
```bash
cd ./sysylib
gcc -c ./sylib.c -o sylib.o
ar rcs libsy.a ./sylib.o
```
2. 



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
- 3.15: 
  - 实现了单层 if-else 语句
  - 添加了 IcmpInst (ieq, ine), BranchInst (cond, no cond)
  - Builder 添加 _is_not 用于在if-else 中记录 cond 中是否有 NOT
    - 感觉还是有问题，应该直接拿到 auto v = visit(exp)，直接判断 v，而不是借助 Builder，因为多层嵌套对 Builder _is_not 的维护有问题
  - 改 `unique_ptr` 为 `*`, 为了方便，以后有需要再调整

目前可实现示例：

```C
int main() {
    int a = 0;
    if (!a) {
        a = 2;
    } else {
        a = 4;
    }
    return a;
}
```

```LLVM
define i32 @main() {
0:     ; block
    %1 = alloca i32
    store i32 0, i32* %1
    %2 = load i32, i32* %1
    %3 = icmp ne i32 0, %2
    br i1 %3, label %5, label %4

4:     ; block
    store i32 2, i32* %1
    br label %6

5:     ; block
    store i32 4, i32* %1
    br label %6

6:     ; block
    %7 = load i32, i32* %1
    ret i32 %7
}
```

- 3.16:
  - if-else 语句的实现
  - visitUnaryExp
  - visitAndExp
  - visitOrExp
  - add detailed comments to them
  - safe_any_cast 实现类型判断，更加安全，避免出现 bad_any_cast
示例：
```C
int main() {
    int a = 1;
    int b = 2;
    if (!a) {
        if (!a) {
            a = 3;
        }
    } else {
        a = 4;
    }

    if (a || !b) {
        a = 5;
    } else {
        a = 6;
    }
    
    if (!a && !b) {
        a = 7;
    }

    return a;
}

```
```LLVM
define i32 @main() {
0:     ; block
    %1 = alloca i32
    store i32 1, i32* %1
    %2 = alloca i32
    store i32 2, i32* %2
    %3 = load i32, i32* %1
    %4 = icmp ne i32 0, %3
    br i1 %4, label %11, label %5

5:     ; block
    %6 = load i32, i32* %1
    %7 = icmp ne i32 0, %6
    br i1 %7, label %9, label %8

8:     ; block
    store i32 3, i32* %1
    br label %10

9:     ; block
    br label %10

10:     ; block
    br label %12

11:     ; block
    store i32 4, i32* %1
    br label %12

12:     ; block
    %13 = load i32, i32* %1
    %14 = icmp ne i32 0, %13
    br i1 %14, label %18, label %15

15:     ; block
    %16 = load i32, i32* %2
    %17 = icmp ne i32 0, %16
    br i1 %17, label %19, label %18

18:     ; block
    store i32 5, i32* %1
    br label %20

19:     ; block
    store i32 6, i32* %1
    br label %20

20:     ; block
    %21 = load i32, i32* %1
    %22 = icmp ne i32 0, %21
    br i1 %22, label %27, label %23

23:     ; block
    %24 = load i32, i32* %2
    %25 = icmp ne i32 0, %24
    br i1 %25, label %27, label %26

26:     ; block
    store i32 7, i32* %1
    br label %28

27:     ; block
    br label %28

28:     ; block
    %29 = load i32, i32* %1
    ret i32 %29

}

```

**2024-3-18**

1. 实现了while, 大家帮忙看看有无bug

**当前效果**

```C
int main(){
    int a=10;
    int b=5;
    while(a&&b){
        a=a-1;
        b=b-1;
        if(a>7 && b>6){
            a=a+1;
        }
    }
    return a;

}

```
```LLVM
;clang产生的代码
define dso_local i32 @main() #0 {
  %1 = alloca i32, align 4
  %2 = alloca i32, align 4
  %3 = alloca i32, align 4
  store i32 0, i32* %1, align 4
  store i32 10, i32* %2, align 4
  store i32 5, i32* %3, align 4
  br label %4

4:                                                ; preds = %25, %0
  %5 = load i32, i32* %2, align 4
  %6 = icmp ne i32 %5, 0
  br i1 %6, label %7, label %10

7:                                                ; preds = %4
  %8 = load i32, i32* %3, align 4
  %9 = icmp ne i32 %8, 0
  br label %10

10:                                               ; preds = %7, %4
  %11 = phi i1 [ false, %4 ], [ %9, %7 ]
  br i1 %11, label %12, label %26

12:                                               ; preds = %10
  %13 = load i32, i32* %2, align 4
  %14 = sub nsw i32 %13, 1
  store i32 %14, i32* %2, align 4
  %15 = load i32, i32* %3, align 4
  %16 = sub nsw i32 %15, 1
  store i32 %16, i32* %3, align 4
  %17 = load i32, i32* %2, align 4
  %18 = icmp sgt i32 %17, 7
  br i1 %18, label %19, label %25

19:                                               ; preds = %12
  %20 = load i32, i32* %3, align 4
  %21 = icmp sgt i32 %20, 6
  br i1 %21, label %22, label %25

22:                                               ; preds = %19
  %23 = load i32, i32* %2, align 4
  %24 = add nsw i32 %23, 1
  store i32 %24, i32* %2, align 4
  br label %25

25:                                               ; preds = %22, %19, %12
  br label %4, !llvm.loop !6

26:                                               ; preds = %10
  %27 = load i32, i32* %2, align 4
  ret i32 %27
}


;./main产生的代码
define i32 @main() {
0:     ; block
    %1 = alloca i32
    store i32 10, i32* %1
    %2 = alloca i32
    store i32 5, i32* %2
    br label %3

3:     ; block
    %4 = load i32, i32* %1
    %5 = icmp ne i32 %4, 0
    br i1 %5, label %6, label %24

6:     ; block
    %7 = load i32, i32* %2
    %8 = icmp ne i32 %7, 0
    br i1 %8, label %9, label %24

9:     ; block
    %10 = load i32, i32* %1
    %11 = sub i32 %10, 1
    store i32 %11, i32* %1
    %12 = load i32, i32* %2
    %13 = sub i32 %12, 1
    store i32 %13, i32* %2
    %14 = load i32, i32* %1
    %15 = icmp sgt i32 %14, 7
    br i1 %15, label %16, label %22

16:     ; block
    %17 = load i32, i32* %2
    %18 = icmp sgt i32 %17, 6
    br i1 %18, label %19, label %22

19:     ; block
    %20 = load i32, i32* %1
    %21 = add i32 %20, 1
    store i32 %21, i32* %1
    br label %23

22:     ; block
    br label %23

23:     ; block
    br label %3

24:     ; block
    %25 = load i32, i32* %1
    ret i32 %25

}


```


- 2024.03.18: 实现了全局变量GlobalVariable (scalar)
  - 还不支持全局变量的常量传播，需要改进
  - 下一步要完整实现 01_var_defn2.c
  - 写一个测试脚本，最好能跟 gcc/llvm 对拍

```C
int g = 3;
int main() {
    int a = g;
    return a;
}
```

```LLVM
@g = global i32 3

define i32 @main() {
0:     ; block
    %1 = load i32, i32* @g
    %2 = alloca i32
    store i32 %1, i32* %2
    %3 = load i32, i32* %2
    ret i32 %3

}

```
- 经证明，二者的lli return echo$?输出的不是一样的

- 但是前者的lli输出和后者的gcc编译可执行文件echo$?的返回值是一样的

- 2024.03.18: 测试脚本 testfunctional.sh

```bash
./test.sh -h
Usage: ./test.sh [-t <test_dir>] [-o <output_dir>] [-s <file>] [-h]
Options:
  -t <test_dir>    Specify the directory containing test files (default: test/steps/)
  -o <output_dir>  Specify the output directory (default: test/.out/)
  -s <file>        Specify a single file to test
  -h               Print this help message

./test.sh -t test/steps/ -o test/.out/ -s test/steps/01_var_defn.c
./test.sh -s ./test/functional/00_main.sy
```


- 2024.03.19: 实现简单函数调用
  - 支持声明-定义，但还没有类型检查
  - 复杂函数，参数更多，类型检查，声明-定义检查需要再做

```c

// ./test.sh -s ./test/steps/func.c

// func.c
int a;

// int func(int q);

int func(int p){
	p = p - 1;
	return p;
}
int main(){
	int b;
	a = 10;
	b = a + 1;
	b = func(a);
	return b;
}

```

```LLVM
; gen.ll
@a = global i32 0

define i32 @func(i32 %0) {
1:     ; block
    %2 = alloca i32
    store i32 %0, i32* %2
    %3 = load i32, i32* %2
    %4 = sub i32 %3, 1
    store i32 %4, i32* %2
    %5 = load i32, i32* %2
    ret i32 %5

}
define i32 @main() {
0:     ; block
    %1 = alloca i32
    store i32 10, i32* @a
    %2 = load i32, i32* @a
    %3 = add i32 %2, 1
    store i32 %3, i32* %1
    %4 = load i32, i32* @a
    %5 = call i32 @func(i32 %4)
    store i32 %5, i32* %1
    %6 = load i32, i32* %1
    ret i32 %6

}

```