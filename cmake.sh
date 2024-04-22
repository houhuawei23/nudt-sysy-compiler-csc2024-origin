#!/bin/bash

# -e: exit on error
# -u: treat unset variables as error
# -x: print commands before executing them
# -E: catch errors in functions and subshells
set -Eeux

# gen target files from template
targets=("generic") #  "riscv"

# gen.py generic_data.yml isa_data.yml output_dir
for target in ${targets[@]}; do
python3 "./src/target/template/gen.py" "./src/target/generic/generic.yml" \
"./src/target/${target}/${target}.yml" "./include/target/${target}/"
done

# python3 ./src/target/template/gen.py ./src/target/generic/generic.yml \
# ./src/target/generic/generic.yml ./include/target/generic/
# python3 ./src/target/template/gen.py ./src/target/generic/generic.yml \
# ./src/target/riscv/riscv.yml ./include/target/riscv/

# cmake config and build
# cmake -S . -B build
# cmake --build build -j16