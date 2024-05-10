#!/bin/bash

clang test.c &> output.txt
grep error output.txt && exit 1
grep "warning: return type of 'main' is not 'int'" output.txt &&
grep "struct Stuff" output.txt
