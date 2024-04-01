./main ./test/steps/00_global_scalar.c > ./gen.ll
llvm-link ./gen.ll ./test/link/link.ll -S -o ./gen.ll
lli ./gen.ll
