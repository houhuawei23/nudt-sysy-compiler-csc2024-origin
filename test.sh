#!/bin/bash
PASS_CNT=0
WRONG_CNT=0
# ERROR_CNT=0
SKIP_CNT=0
ALL_CNT=0
WRONG_FILES=()

# Color setting
RED=$(tput setaf 1)
GREEN=$(tput setaf 2)
YELLOW=$(tput setaf 3)
BLUE=$(tput setaf 4)
MAGENTA=$(tput setaf 5)
CYAN=$(tput setaf 6)
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
    echo "  -t <test_dir/file>  Specify the directory containing test files (default: test/steps/) or single file"
    echo "  -o <output_dir>     Specify the output directory (default: test/.out/)"
    echo "  -r <result_file>    Specify the file to store the test results (default: test/result.txt)"
    echo "  -h                  Print this help message"
}

# Parse command line arguments
while getopts ":ht:o:s:r:" opt; do
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

echo "test_path      ${test_path} " >${result_file}
echo "output_dir     ${output_dir} " >>${result_file}
echo "result_file    ${result_file} " >>${result_file}

echo "" >>${result_file}

sysylib_funcs=("getint" "getch" "getarray" "getfloat" "getfarray" "putint" "putch" "putarray" "putfloat" "putfarray" "putf" "starttime" "stoptime")

function check_key_in_file() {
    file="$1"
    for func in "${sysylib_funcs[@]}"; do
        if grep -q "$func" "$file"; then
            return 1 # 文件中包含关键字，返回 1
        fi
    done
    return 0 # 文件中未找到关键字，返回 0
}

# define a function that test one file
function run_test() {
    single_file="$1"
    output_dir="$2"
    result_file="$3"

    check_key_in_file "$single_file"
    if [ $? -eq 1 ]; then
        echo "${CYAN}[SKIP]${RESET} ${single_file}"
        SKIP_CNT=$((SKIP_CNT + 1))
        return 0
    fi

    if [ -f "$single_file" ]; then

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
    # traverse all the *.c and *.sy file
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

    ALL_CNT=$((PASS_CNT + WRONG_CNT + SKIP_CNT))
    echo "${GREEN}PASS ${RESET}: ${PASS_CNT}"
    echo "${RED}WRONG${RESET}: ${WRONG_CNT}"
    echo "${BLUE}SKIP ${RESET}: ${SKIP_CNT}"
    echo "${YELLOW}ALL  ${RESET}: ${ALL_CNT}"
fi
