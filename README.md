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

## code到AST的分析

antlr4是一个编译器前端生成工具，可以生成多种目标语言的前端。本项目生成的目标语言是C++。
```shell
# antlr4compile.sh
cd src
java -jar ../antlr/antlr-4.12.0-complete.jar \
    -Dlanguage=Cpp -no-listener -visitor \
    SysY.g4
cd ..
cmake -S . -B build
cmake --build build
cd build
make
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
