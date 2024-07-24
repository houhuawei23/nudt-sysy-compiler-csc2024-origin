#!/bin/bash

pip3 install -r requirements.txt

sudo apt-get update
sudo apt-get install -y build-essential uuid-dev libutfcpp-dev pkg-config make git cmake openjdk-11-jre

sudo apt-get install -y clang clang-format ninja-build llvm-14 llvm-14-dev 

sudo apt install -y qemu-user gcc-riscv64-linux-gnu g++-12-riscv64-linux-gnu

sudo apt install gdb-multiarch 

sudo apt install tldr valgrind

sudo ln -s /usr/bin/llvm-link-14 /usr/bin/llvm-link

# qemu-system-* is for system emulation

ln -s /usr/bin/llvm-link-14 /usr/bin/llvm-link

sudo apt install libantlr4-runtime-dev antlr4 libgtest-dev default-jdk
