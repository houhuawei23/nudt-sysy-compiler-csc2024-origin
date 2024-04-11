#!/bin/bash
PASS_CNT=0
WRONG_CNT=0
ALL_CNT=0
WRONG_FILES=()

# Color setting
RED=$(tput setaf 1)
GREEN=$(tput setaf 2)
YELLOW=$(tput setaf 3)
RESET=$(tput sgr0)

# Default values
test_path="test/local_test"
output_dir="test/.out"
single_file=""
result_file="test/.out/result.txt"

TIMEOUT=3

# Function to print usage information
usage() {
    echo "Usage: $0 [-t <test_path>] [-o <output_dir>] [-r <result_file>] [-r <result_file>][-h]"
    echo "Options:"
    echo "  -t <test_path>  Specify the directory containing test files or single file (default: test/local_test)"
    echo "  -o <output_dir>     Specify the output directory (default: test/.out/)"
    echo "  -r <result_file>    Specify the file to store the test results (default: test/result.txt)"
    echo "  -h                  Print this help message"
}

# Parse command line arguments
while getopts ":ht:o:r:" opt; do
    case $opt in
    h)
        usage
        exit 0
        ;;
    t)
        test_path="$OPTARG"
        ;;
    o)
        output_dir="$OPTARG"
        ;;
    r)
        result_file="$OPTARG"
        ;;
    :)
        echo "Option -$OPTARG requires an argument." >&2
        usage
        exit 1
        ;;
    ?)
        echo "Invalid option: -$OPTARG" >&2
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

echo "test_path      ${test_path} "   >>${result_file}
echo "output_dir     ${output_dir} "  >>${result_file}
echo "result_file    ${result_file} " >>${result_file}

echo "" >>${result_file}

# define a function that test one file
function run_test() {
    single_file="$1"
    output_dir="$2"
    result_file="$3"

    if [ -f "$single_file" ]; then
        in_file="${single_file%.*}.in"

        # llvm compiler
        cp "$single_file" "${output_dir}/test.c"
        clang --no-warnings -emit-llvm -S "${output_dir}/test.c" -o "${output_dir}/llvm.ll" -O0
        llvm-link --suppress-warnings ./test/link/link.ll "${output_dir}/llvm.ll" -S -o "${output_dir}/llvm_linked.ll"
        if [ -f "$in_file" ]; then
            lli "${output_dir}/llvm_linked.ll" > "${output_dir}/llvm.out" < "${in_file}"
        else 
            lli "${output_dir}/llvm_linked.ll" > "${output_dir}/llvm.out"
        fi
        llvmres=$?
        # llvm compiler end

        # sys-compiler
        ./main "$single_file" >"${output_dir}/gen.ll"
        llvm-link --suppress-warnings ./test/link/link.ll "${output_dir}/gen.ll" -S -o "${output_dir}/gen_linked.ll"
        if [ $? != 0 ]; then
            echo "link error"
            echo "${RED}[WRONG]${RESET} ${single_file}"
            echo "    res (${res}), llvmres (${llvmres})"
            echo "[WRONG] ${single_file}" >>${result_file}
            WRONG_CNT=$((WRONG_CNT + 1))
            WRONG_FILES+=($single_file)
            return 1
        fi

        if [ -f "$in_file" ]; then
            timeout $TIMEOUT lli "${output_dir}/gen_linked.ll" > "${output_dir}/gen.out" < "${in_file}"
            if [ $? == 124 ]; then
                echo "link timeout"
                echo "${RED}[WRONG]${RESET} ${single_file}"
                echo "    res (${res}), llvmres (${llvmres})"
                echo "[WRONG] ${single_file}" >>${result_file}
                WRONG_CNT=$((WRONG_CNT + 1))
                WRONG_FILES+=($single_file)
                return 1
            fi
            lli "${output_dir}/gen_linked.ll" > "${output_dir}/gen.out" < "${in_file}"
        else 
            timeout $TIMEOUT lli "${output_dir}/gen_linked.ll" > "${output_dir}/gen.out"
            if [ $? == 124 ]; then
                echo "link timeout"
                echo "${RED}[WRONG]${RESET} ${single_file}"
                echo "    res (${res}), llvmres (${llvmres})"
                echo "[WRONG] ${single_file}" >>${result_file}
                WRONG_CNT=$((WRONG_CNT + 1))
                WRONG_FILES+=($single_file)
                return 1
            fi
            lli "${output_dir}/gen_linked.ll" > "${output_dir}/gen.out"
        fi
        res=$?
        # sys-compiler end

        diff "${output_dir}/gen.out" "${output_dir}/llvm.out" > "/dev/null"
        diff_res=$?

        if [ $res != $llvmres ] || [ $diff_res != 0 ]; then
            echo "${RED}[WRONG]${RESET} ${single_file}"
            echo "    res (${res}), llvmres (${llvmres})"
            echo "[WRONG] ${single_file}" >>${result_file}
            echo "  [WRONG]: res (${res}), llvmres (${llvmres})" >>${result_file}
            WRONG_CNT=$((WRONG_CNT + 1))
            WRONG_FILES+=($single_file)
        else
            echo "${GREEN}[CORRECT]${RESET} ${single_file}"
            echo "[CORRECT] ${single_file}" >>${result_file}
            PASS_CNT=$((PASS_CNT + 1))
        fi
    else
        echo "File not found: $single_file"
        exit 1
    fi
}

# if test_path is a file
if [ -f "$test_path" ]; then
    run_test "$test_path" "$output_dir" "$result_file"
fi
file_types=("*.c" "*.sy")
# if test_path is a directory

if [ -d "$test_path" ]; then
    for file_type in "${file_types[@]}"; do
        for file in "${test_path}"/${file_type}; do
            if [ ! -f "${file}" ]; then
                break
            else
                run_test "$file" "$output_dir" "$result_file"
            fi
        done

    done
    echo "${RED}[WRONG]${RESET} files:"
    for file in "${WRONG_FILES[@]}"; do
        echo "${file}"
    done

    ALL_CNT=$((PASS_CNT + WRONG_CNT))
    echo "${GREEN}PASS ${RESET}: ${PASS_CNT}"
    echo "${RED}WRONG${RESET}: ${WRONG_CNT}"
    echo "${YELLOW}ALL  ${RESET}: ${ALL_CNT}"
fi
