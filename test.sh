#!/bin/bash

# set -Eeuo pipefail
# # -e          设置脚本在执行出错时终止
# # -u          设置遇到不存在的变量就报错
# # -x          先打印命令，再执行
# # -o pipefail 管道符子命令出错终止
# # -E          set -e 使得函数内部错误不被trap捕获，-E使其再被捕获

# Color setting
RED=$(tput setaf 1)
GREEN=$(tput setaf 2)
YELLOW=$(tput setaf 3)
RESET=$(tput sgr0)

# Default values
test_dir="test/steps"
output_dir="test/.out"
single_file=""
result_file="test/.out/result.txt"

# Function to print usage information
usage() {
    echo "Usage: $0 [-t <test_dir>] [-o <output_dir>] [-s <file>] [-r <result_file>][-h]"
    echo "Options:"
    echo "  -t <test_dir>     Specify the directory containing test files (default: test/steps/)"
    echo "  -o <output_dir>   Specify the output directory (default: test/.out/)"
    echo "  -s <file>         Specify a single file to test"
    echo "  -r <result_file>  Specify the file to store the test results (default: test/result.txt)"
    echo "  -h                Print this help message"
}

# Parse command line arguments
while getopts ":ht:o:s:r:" opt; do
    case $opt in
    h)
        usage
        exit 0
        ;;
    t)
        test_dir="$OPTARG"
        ;;
    o)
        output_dir="$OPTARG"
        ;;
    s)
        single_file="$OPTARG"
        ;;
    r)
        result_file="$OPTARG"
        ;;
    \?)
        echo "Invalid option: -$OPTARG" >&2
        usage
        exit 1
        ;;
    :)
        echo "Option -$OPTARG requires an argument." >&2
        usage
        exit 1
        ;;
    esac
done

shift $((OPTIND - 1))

# Ensure output directory exists
if [ ! -d "$output_dir" ]; then
    mkdir -p "$output_dir"
fi

echo "test_dir       ${test_dir} " >${result_file}
echo "output_dir     ${output_dir} " >>${result_file}
echo "result_file    ${result_file} " >>${result_file}
if [ -n "$single_file" ]; then
    echo "single_file    ${single_file} " >>${result_file}
fi
echo "" >>${result_file}
# echo "single_file ${single_file} " > ${single_file}

# Main loop
if [ -n "$single_file" ]; then
    # Test only the specified file if -s option is provided
    if [ -f "$single_file" ]; then
        echo "${YELLOW}TEST${RESET}: ${YELLOW} ${single_file} ${RESET}"
        echo "TEST:  ${single_file} " >>${result_file}
        ./main "$single_file" >"${output_dir}/gen.ll"
        lli "${output_dir}/gen.ll"
        res=$?

        if [[ "$single_file" == *.c ]]; then
            clang -emit-llvm -S "$single_file" -o "${output_dir}/llvm.ll"
            lli "${output_dir}/llvm.ll"
            llvmres=$?
        elif [[ "$single_file" == *.sy ]]; then
            cp "$single_file" "${output_dir}/test.c"
            clang -emit-llvm -S "${output_dir}/test.c" -o "${output_dir}/llvm.ll"
            lli "${output_dir}/llvm.ll"
            llvmres=$?
        else
            echo "Unsupported file type: ${single_file}"
        fi

        if [ $res != $llvmres ]; then
            echo "${RED}xx WRONG RESULT${RESET}: res (${res}), llvmres (${llvmres})"
            echo "xx WRONG RESULT: res (${res}), llvmres (${llvmres})" >>${result_file}
        else
            echo "${GREEN}== CORRECT ==${RESET}"
            echo "== CORRECT ==" >>${result_file}
        fi
    else
        echo "File not found: $single_file"
        exit 1
    fi
else
    # Test all files in the test directory
    for file in "${test_dir}"/*.c "${test_dir}"/*.sy; do
        if [ -f "$file" ]; then
            echo "${YELLOW}TEST${RESET}: ${YELLOW} ${file} ${RESET}"
            echo "TEST:  ${file} " >>${result_file}
            ./main "$file" >"${output_dir}/gen.ll"
            lli "${output_dir}/gen.ll"
            res=$?

            # clang -emit-llvm dont support *.sy file, so create tmp *.c file
            if [[ "$file" == *.c ]]; then
                clang -emit-llvm -S "$file" -o "${output_dir}/llvm.ll"
                lli "${output_dir}/llvm.ll"
                llvmres=$?
            elif [[ "$file" == *.sy ]]; then
                cp "$file" "${output_dir}/test.c"
                clang -emit-llvm -S "${output_dir}/test.c" -o "${output_dir}/llvm.ll"
                lli "${output_dir}/llvm.ll"
                llvmres=$?
            else
                echo "Unsupported file type: ${single_file}"
            fi

            if [ $res != $llvmres ]; then
                echo "${RED}xx WRONG RESULT${RESET}: res (${res}), llvmres (${llvmres})"
                echo "xx WRONG RESULT: res (${res}), llvmres (${llvmres})" >>${result_file}
            else
                echo "${GREEN}== CORRECT ==${RESET}"
                echo "== CORRECT ==" >>${result_file}
            fi
        fi
    done
fi
