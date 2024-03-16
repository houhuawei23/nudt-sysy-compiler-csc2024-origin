./main ./test/steps/00_global_scalar.c > test.ll
lli test.ll
echo $?
