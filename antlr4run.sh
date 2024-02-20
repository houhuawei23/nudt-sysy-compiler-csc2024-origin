cd src
java -jar ../antlr/antlr-4.12.0-complete.jar \
    -Dlanguage=Cpp -no-listener -visitor \
    SysY.g4
cd ..
cmake -S . -B build
cmake --build build
cd build
make
