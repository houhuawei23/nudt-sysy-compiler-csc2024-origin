#!/bin/bash

## Example usage:

# ./qemu.sh ./test/.local_test/local_test.c

## Then waiting for the gdb terminal connected to qemu
## open another terminal and run:
# ./gdb.sh ./gen.o

set -u
set -e

infile=$1
asmfile="./gen.s"
outfile="./gen.o"

./main -f $infile -S -o $asmfile

riscv64-linux-gnu-gcc -ggdb -static -march=rv64gc -mabi=lp64d -mcmodel=medlow \
 -o "${outfile}" "${asmfile}" 

sudo qemu-riscv64 -g 1235 $outfile