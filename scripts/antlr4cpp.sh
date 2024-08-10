#!/bin/bash
main_path=$(pwd)

cd $main_path/src/antlr4

java -jar $main_path/antlr/antlr-4.12.0-complete.jar \
    -Dlanguage=Cpp -no-listener -visitor \
    SysY.g4 -o $main_path/src/.antlr4cpp


java -jar $main_path/antlr/antlr-4.12.0-complete.jar \
    SysY.g4 -o $main_path/src/antlr/.antlr

# cd $main_path/src/antlr4/.antlr
# javac SysY*.java 