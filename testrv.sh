#!/bin/bash
# mkdir in out sy; mv *.sy sy; mv *.in in; mv *.out out; 

#set -u
#set -x

TIMEOUT=0.5

PASS_CNT=0
WRONG_CNT=0
ALL_CNT=0
TIMEOUT_CNT=0
SKIP_CNT=0

WRONG_FILES=()
TIMEOUT_FILES=()

PASSES=()

OPT_LEVEL="-O0"
LOG_LEVEL="-L0"

# Color setting
RED=$(tput setaf 1)
GREEN=$(tput setaf 2)
YELLOW=$(tput setaf 3)
RESET=$(tput sgr0)
CYAN=$(tput setaf 6)

EC_MAIN=1
EC_RISCV_GCC=2
EC_LLI=3
EC_TIMEOUT=124

sysylib_funcs=(
#    "getint" "getch" "getarray" "getfloat" "getfarray"
#    "putint" "putch" "putarray" "putfloat" "putfarray"
#    "putf" "starttime" "stoptime"
)
skip_keywords=(
    # "\["
    # "if"
    # "while"
)
function check_key_in_file() {
    file="$1"
    for func in "${sysylib_funcs[@]}"; do
        if grep -q "$func" "$file"; then
            return 1 
        fi
    done

    for keyword in "${skip_keywords[@]}"; do
        if grep -q "$keyword" "$file"; then
        return 1
	fi
    done

    return 0 
}

test_path="test/local_test"
output_dir="test/.out"
single_file=""
result_file="test/.out/result.txt"


usage() {
    echo "Usage: $0 [-t <test_path>] [-o <output_dir>] [-r <result_file>] [-r <result_file>][-h]"
    echo "Options:"
    echo "  -t <test_path>  Specify the directory containing test files or single file (default: test/local_test)"
    echo "  -o <output_dir>     Specify the output directory (default: test/.out/)"
    echo "  -r <result_file>    Specify the file to store the test results (default: test/result.txt)"
    echo "  -h                  Print this help message"
}

SHORT="h,t:,o:,r:,p:,O:,L:"
LONG="help,test_path:,output_dir:,result_file:,pass:opt_level:,log_level:"
OPTS=$(getopt --options $SHORT --longoptions $LONG --name "$0" -- "$@")
if [ $? -ne 0 ]; then
    echo "Error parsing command line arguments" >&2
    usage
    exit 1
fi
eval set -- "$OPTS"

while true; do
    case "$1" in
    -h | --help)
        usage
        exit 0
        ;;
    -t | --test_path)
        test_path="$2"
        shift 2
        ;;
    -o | --output_dir)
        output_dir="$2"
        shift 2
        ;;
    -r | --result_file)
        result_file="$2"
        shift 2
        ;;
    -p | --pass)
        PASSES+=("$2")
        shift 2
        ;;
    -O | --opt_level)
        OPT_LEVEL="-O$2"
        shift 2
        ;;
    -L | --log_level)
        LOG_LEVEL="-L$2"
        shift 2
        ;;
    --)
        shift
        break
        ;;
    *)
        echo "Invalid option: $1" >&2
        usage
        exit 1
        ;;
    esac
done

# Handle remaining arguments (non-option arguments)
# Use $@ or $1, $2, etc. depending on the specific needs

PASSES_STR=$(
    IFS=" "
    echo "${PASSES[*]}"
)

# Ensure output directory exists
if [ ! -d "$output_dir" ]; then
    mkdir -p "$output_dir"
fi

sy_h="./test/link/sy.h"
sy_c="./test/link/link.c"

function run_gcc_test() {
    local single_file="$1"
    local output_dir="$2"
    local result_file="$3"
    local file_name=$(basename $single_file)
    local in_file="${in_dir}/${file_name%.*}.in"
    echo $in_file
    local gcc_c="${output_dir}/gcc_test.c"
    local gcc_s="${output_dir}/gcc.s"
    local gcc_o="${output_dir}/gcc.o"

    gcc_out="${output_dir}/gcc_out"
    if [ -f "${gcc_c}" ]; then
        rm "${gcc_c}"
    fi
    touch "${gcc_c}"
    cat $sy_h > ${gcc_c}
    cat "${single_file}" >>"${gcc_c}"

    gcc "${gcc_c}" "${sy_c}" -o "${gcc_o}" -O0
    if [ -f "${in_file}" ]; then
       ./$gcc_o < ${in_file} >"${gcc_out}"
    else
       ./$gcc_o >"${gcc_out}"
    fi
    local gcc_res=$?
    # echo "gcc_res: $gcc_res"
    return $gcc_res
}

function run_compiler_test() {
    local single_file="$1"
    local output_dir="$2"
    local result_file="$3"

    local file_name=$(basename "$single_file")
    local in_file="${in_dir}/${file_name%.*}.in"


    local gen_s="${asm_dir}/${file_name%.*}.s"
    if [ ! -f $gen_s ]; then
	return -1
    fi

    local gen_o="${output_dir}/gen.o"

    gen_out="${output_dir}/gen_out"

    # for test
    gcc -march=rv64gc "${gen_s}" "${sy_c}" -o "${gen_o}"
    if [ $? != 0 ]; then
	return -1
    fi
    if [ -f "${in_file}" ]; then
        ./"${gen_o}" <${in_file} >"${gen_out}"
    else
        ./"${gen_o}" >"${gen_out}"

    fi
    local compiler_res=$?
    # echo "compiler_res: $compiler_res"
    return $compiler_res
}
function run_test_asm() {
    local single_file="$1"
    local output_dir="$2"
    local result_file="$3"

    check_key_in_file "$single_file"
    # echo "check_key_in_file res: $?"
    if [ $? != 0 ]; then
        echo "${CYAN}[SKIP]${RESET} ${single_file}"
        SKIP_CNT=$((SKIP_CNT + 1))
        return 0
    fi

    if [ -f "$single_file" ]; then
        echo "${YELLOW}[Testing]${RESET} $single_file"

        run_gcc_test "${single_file}" "${output_dir}" "${result_file}"
        local gccres=$?

        run_compiler_test "${single_file}" "${output_dir}" "${result_file}"
        local res=$?

        diff "${gen_out}" "${gcc_out}" >"${output_dir}/diff.out"
        local diff_res=$?
        # diff res or diff stdout
        echo "[RESULT] res (${RED}${res}${RESET}), gccres (${RED}${gccres}${RESET})"

        if [ ${res} != ${gccres} ] || [ ${diff_res} != 0 ]; then

            if [ ${res} == ${EC_MAIN} ]; then
                echo "${RED}[MAIN ERROR]${RESET} ${single_file}"
            elif [ ${res} == ${EC_RISCV_GCC} ]; then
                echo "${RED}[RISCV-GCC ERROR]${RESET} ${single_file}"
            elif [ ${res} == ${EC_LLI} ]; then
                echo "${RED}[LLI ERROR]${RESET} ${single_file}"
            elif [ ${res} == ${EC_TIMEOUT} ]; then
                echo "${RED}[TIMEOUT]${RESET} ${single_file}"
                echo "[TIMEOUT] ${single_file}" >>${result_file}
            else
                echo "${RED}[WRONG RES]${RESET} ${single_file}"
                echo "[WRONG RES] ${single_file}" >>${result_file}
                echo "  [WRONG RES]: res (${res}), gccres (${gccres})" >>${result_file}
            fi

            if [ ${res} == ${EC_TIMEOUT} ]; then
                TIMEOUT_CNT=$((TIMEOUT_CNT + 1))
                TIMEOUT_FILES+=(${single_file})
            else
                WRONG_CNT=$((WRONG_CNT + 1))
                WRONG_FILES+=(${single_file})
            fi
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

file_types=("*.c" "*.sy")
file_name=$(basename $test_path)
# if test_path is a file
if [ -f "$test_path" ]; then
    test_dir="$(dirname "$test_path")/.." 
    sy_dir="${test_dir}/sy"
    in_dir="${test_dir}/in"
    out_dir="${test_dir}/out"
    asm_dir="${test_dir}/asm"
    # run_gcc_test "$test_path" "$output_dir" "$result_file" $in_dir

    # run_compiler_test "$test_path" "$output_dir" "$result_file"
    run_test_asm "$test_path" "$output_dir" "$result_file"
    echo "${GREEN}OPT PASSES${RESET}: ${PASSES_STR}"
fi

test_dir=$test_path
sy_dir="${test_dir}/sy"
in_dir="${test_dir}/in"
out_dir="${test_dir}/out"
asm_dir="${test_dir}/asm"
# echo $in_dir
# if test_path is a directory
# echo $sy_dir
if [ -d "$sy_dir" ]; then
    for file_type in "${file_types[@]}"; do
        for file in "${sy_dir}"/${file_type}; do
            echo $file
	    if [ ! -f "${file}" ]; then
                break
            else
		filename=$(basename $file)
		echo $filename
                # run_gcc_test "${file}" "${output_dir}" "${result_file}" 
    		run_test_asm "$file" "$output_dir" "$result_file"
            fi
        done

    done

    echo "====  RESULT  ===="

    echo "${RED}[WRONG]${RESET} files:"
    for file in "${WRONG_FILES[@]}"; do
        echo "${file}"
    done
    echo "${RED}[TIMEOUT]${RESET} files:"
    for file in "${TIMEOUT_FILES[@]}"; do
        echo "${file}"
    done

    echo "====   INFO   ===="
    echo "PASSES: ${PASSES_STR}"

    ALL_CNT=$((PASS_CNT + WRONG_CNT + SKIP_CNT))
    echo "${GREEN}PASS ${RESET}: ${PASS_CNT}"
    echo "${RED}WRONG${RESET}: ${WRONG_CNT}"
    echo "${RED}TIMEOUT${RESET}: ${TIMEOUT_CNT}"
    echo "${CYAN}SKIP ${RESET}: ${SKIP_CNT}"
    echo "${YELLOW}ALL  ${RESET}: ${ALL_CNT}"
fi
