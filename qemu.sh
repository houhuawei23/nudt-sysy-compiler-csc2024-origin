#!/bin/bash

## Example usage:

# ./qemu.sh ./test/.local_test/local_test.c

## Then waiting for the gdb terminal connected to qemu
## open another terminal and run:
# ./gdb.sh ./gen.o

set -u
set -e
# set -x
# infile=$1
asmfile="./test/.out/gen.s"
outfile="./gen.o"
# memset_s="./test/link/memset.s"
compiler_path="./compiler"

# ./compiler -f $infile -S -o $asmfile

riscv64-linux-gnu-gcc -ggdb -static -march=rv64gc -mabi=lp64d -mcmodel=medlow \
 -o "${outfile}" "${asmfile}" ./test/link/link.c

sudo qemu-riscv64 -L /usr/riscv64-linux-gnu/ -g 1235 $outfile
