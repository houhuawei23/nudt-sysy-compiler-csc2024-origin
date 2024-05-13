#!/bin/bash

set -u
set -e

## Example usage:

# ./qemu.sh ./test/.local_test/local_test.c

## Then waiting for the gdb terminal connected to qemu
## open another terminal and run:
# ./gdb.sh ./gen.o


file=$1

gdb-multiarch -x ./config/gdb.conf $file