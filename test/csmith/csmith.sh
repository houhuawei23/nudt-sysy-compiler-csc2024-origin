#!/bin/bash
export PATH=$PATH:/usr/csmith/bin
csmith --no-pointers --quiet --no-packed-struct --no-unions --no-volatiles \
--no-volatile-pointers --no-const-pointers --no-builtins --no-jumps \
--no-bitfields --no-argc --no-structs --no-longlong --no-uint8 --no-math64 \
--max-funcs 2 --max-block-depth 2 --max-expr-complexity 3  -o test.c
