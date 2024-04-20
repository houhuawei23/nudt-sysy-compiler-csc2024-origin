#!/bin/bash

# -e: exit on error
# -u: treat unset variables as error
# -x: print commands before executing them
# -E: catch errors in functions and subshells
set -Eeux

# gen target files from template
targets=("riscv" "generic") #  

for target in ${targets[@]}; do
    python3 ./src/target/template/gen.py ./src/target/${target}/${target}.yml ./include/target/${target}/
done

# python3 ./src/target/template/gen.py ./src/target/generic/generic.yml \
# ./src/target/generic/generic.yml ./include/target/generic/

# cmake config and build
cmake -S . -B build
cmake --build build -j16