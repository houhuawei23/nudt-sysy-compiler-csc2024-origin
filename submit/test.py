import os
import sys
import subprocess
import shutil

from utils import check_args

import colorama
from colorama import Fore, Style


colorama.init(autoreset=True)  # Initializes colorama and autoresets color


# python test.py compiler_path tests_path output_asm_path output_exe_path
# python ./submit/test.py ./compiler ./test/ ./.tmp/asm ./.tmp/exe

compiler_path = sys.argv[1]
tests_path = sys.argv[2]  #
output_asm_path = sys.argv[3]
output_exe_path = sys.argv[4]


if not check_args():
    sys.exit(1)


stack_size = 128 << 20  # 128M

qemu_command = f"qemu-riscv64 -L /usr/riscv64-linux-gnu/ -cpu rv64,zba=true,zbb=true -s {stack_size} -D /dev/stderr".split()

gcc_ref_command = "gcc -x c++ -O3 -DNDEBUG -march=native -fno-tree-vectorize -s -funroll-loops -ffp-contract=on -w ".split()

clang_ref_command = "clang -Qn -O3 -DNDEBUG -emit-llvm -fno-slp-vectorize -fno-vectorize -mllvm -vectorize-loops=false -S -ffp-contract=on -w ".split()

qemu_gcc_ref_command = "riscv64-linux-gnu-gcc-12 -O2 -DNDEBUG -march=rv64gc -mabi=lp64d -mcmodel=medlow -ffp-contract=on -w ".split()


sysy_runtime = tests_path + "/sysy/sylib.c"
sysy_header = tests_path + "/sysy/sylib.h"


def test(testname: str, path: str, suffix: str, tester):
    """
    test files with suffix in path with tester.
    for f in all files in path with suffix:
      tester(f)
    """
    print(f"Testing {testname}...")
    print(f"Test files with suffix {suffix} in {path}")
    test_list = []
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            if f.endswith(suffix):
                test_list.append(os.path.join(dirpath, f))
    cnt = 0
    failed_list = []
    test_list.sort()
    for src in test_list:
        cnt += 1
        print(f"Test {cnt}/{len(test_list)}: {src}")
        try:
            if tester(src) is not False:
                print("Test passed")
                continue
        except Exception as e:
            print(f"Test failed: {e}")
        print("Test failed")
        failed_list.append(src)

    return len(test_list), len(failed_list)



from enum import Enum


class ResultType(Enum):
    PASSED = 0
    RUN_COMPILER_FAILED = 1
    LINK_EXECUTABLE_FAILED = 2
    RUN_EXECUTABLE_FAILED = 3
    OUTPUT_MISMATCH = 4


colorMap = {
    ResultType.PASSED: Fore.GREEN,
    ResultType.RUN_COMPILER_FAILED: Fore.RED,
    ResultType.LINK_EXECUTABLE_FAILED: Fore.RED,
    ResultType.RUN_EXECUTABLE_FAILED: Fore.RED,
    ResultType.OUTPUT_MISMATCH: Fore.RED,
}



class TestResult:
    # list of (src, completed_process)
    cases_result = {key: [] for key in ResultType}

    def __init__(self, test_name):
        self.test_name = test_name

    def print_result_overview(self):
        print(Fore.YELLOW + f"Test {self.test_name}:")
        all = self.all_cases()
        passed = self.cases_result[ResultType.PASSED]

        print(
            f"Total: {len(all)}, Passed: {len(passed)}, Failed: {len(all)-len(passed)}"
        )
        for type in ResultType:
            print(colorMap[type] + f"{type.name}: {len(self.cases_result[type])}")
        print()
        for type in ResultType:
            if type == ResultType.PASSED:
                continue
            if len(self.cases_result[type]) == 0:
                continue
            self.print_result(type)

    def all_cases(self):
        all = []
        for key in ResultType:
            all.extend(self.cases_result[key])
        return all

    def print_result(self, type: ResultType):
        print(f"Test {self.test_name}" + colorMap[type] + f" {type.name}:")
        for src, process in result.cases_result[type]:
            print(Fore.YELLOW + f"test: {src}")
            print(f"returncode: {process.returncode}")
            print("stdout:")
            print(repr(process.stdout[:100]))
            print("stderr:")
            print(repr(process.stderr[:100]))

    def save_result(self, filename: str):
        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"Test {self.test_name}")
            for type in ResultType:
                if type == ResultType.PASSED:
                    continue
                f.write(f"\n{type.name}:\n")
                for src, process in self.cases_result[type]:
                    f.write(f"test: {src}\n")
                    f.write(f"returncode: {process.returncode}\n")
                    f.write("stdout:\n")
                    f.write(repr(process.stdout[:100]))
                    f.write("\n")
                    f.write("stderr:\n")
                    f.write(repr(process.stderr[:100]))
                    f.write("\n")


result = TestResult("SysY compiler functional")


def run_compiler(
    src, target, output, opt_level=0, debug_level=0, emit_ir=False, timeout=1
):
    """
    ./compiler -S -o output src
    ./compiler -S -o output src -O1
    """
    command = [compiler_path, "-S", "-o", output, src, "-O1"]
    process = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    return process


def basename(filename: str):
    """
    basename("a/b/c.txt") => "a/b/c"
    """
    return os.path.splitext(filename)[0]


def link_executable(
    src: str, target: str, output: str, runtime=sysy_runtime, timeout=1
):
    """
    riscv64-linux-gnu-gcc-12
    """
    command = qemu_gcc_ref_command + ["-o", output, runtime, src]
    process = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    return process


def compare_output_with_standard_file(
    standard_filename: str, output: str, returncode: int
):
    if len(output) != 0 and not output.endswith("\n"):
        output += "\n"
    output += str(returncode) + "\n"

    with open(standard_filename, encoding="utf-8", newline="\n") as f:
        standard_answer = f.read()
    if not standard_answer.endswith("\n"):
        standard_answer += "\n"

    standard_answer = standard_answer.replace("\r\n", "\n")

    if output != standard_answer:
        print(Fore.RED + " Output mismatch")
        print("--------")
        print("output:")
        print(repr(output[:100]))
        print("--------")
        print("stdans:")
        print(repr(standard_answer[:100]))
        print("--------")
        return False
    return True


def compare_and_parse_perf(src: str, out: subprocess.CompletedProcess):
    """
    compare and parse perf output
    """
    output_file = basename(src) + ".out"
    if not compare_output_with_standard_file(output_file, out.stdout, out.returncode):
        raise RuntimeError("Output mismatch")

    for line in out.stderr.splitlines():
        if line.startswith("insns:"):
            used = int(line.removeprefix("insns:").strip())
            if "performance" in src:
                print(f" {used}", end="")
            return used

    for line in out.stderr.splitlines():
        if line.startswith("TOTAL:"):
            perf = line.removeprefix("TOTAL: ").split("-")
            used = (
                float(perf[0][:-1]) * 3600
                + float(perf[1][:-1]) * 60
                + float(perf[2][:-1])
                + float(perf[3][:-2]) * 1e-6
            )
            if "performance" in src:
                print(f" {used:.6f}", end="")
            return max(1e-6, used)

    raise RuntimeError("No performance data")


def run_executable(command, src, timeout=1):
    input_file = basename(src) + ".in"
    if os.path.exists(input_file):
        with open(input_file, "r", encoding="utf-8") as f:
            out = subprocess.run(
                command, stdin=f, capture_output=True, text=True, timeout=timeout
            )
    else:
        out = subprocess.run(command, capture_output=True, text=True, timeout=timeout)

    # time_used = compare_and_parse_perf(src, out)
    # return time_used

    res = compare_output_with_standard_file(
        basename(src) + ".out", out.stdout, out.returncode
    )
    return res, out


# import time
from datetime import datetime

now = datetime.now()

dt_string = now.strftime("%Y_%m_%d_%H:%M")


class Test:
    def __init__(self, target: str, year: str, test_kind: str, timeout=5):
        self.target = target
        self.year = year
        self.test_kind = test_kind
        self.result = TestResult(f"SysY compiler {year} {test_kind}")
        self.timeout = timeout

    def run(self):
        print(Fore.RED + f"Testing {self.year} {self.test_kind}...")
        year_kind_path = os.path.join(tests_path, self.year, self.test_kind)
        testnum, failednum = self.test_beta(
            year_kind_path,
            ".sy",
            lambda x: self.sysy_compiler_qemu(x, self.target),
        )
        self.result.print_result_overview()
        self.result.save_result(f"./{self.year}_{self.test_kind}_{dt_string}.md")

    def run_single_case(self, test_case_rel_path: str):
        """
        test.run_single_case("2023/functional/04_arr_defn3.sy")
        """
        test_case_path = os.path.join(tests_path, test_case_rel_path)
        self.sysy_compiler_qemu(test_case_path, self.target)
        # self.result.print_result_overview()
        for type in ResultType:
            if len(self.result.cases_result[type]) > 0:
                self.result.print_result(type)

    def sysy_compiler_qemu(self, src: str, target: str):
        if os.path.exists(src) is False:
            print(f"Test file not found: {src}")
            return False
        filename = os.path.basename(src)
        raw_name = os.path.splitext(filename)[0]
        output_exe = os.path.join(output_exe_path, raw_name)
        output_asm = os.path.join(output_asm_path, raw_name + ".s")

        try:
            run_compiler_process = run_compiler(
                src, target, output_asm, timeout=self.timeout
            )
        except subprocess.TimeoutExpired:
            print(Fore.RED + f"Test {src} run_compiler timeout")
            self.result.cases_result[ResultType.RUN_COMPILER_FAILED].append(
                (
                    src,
                    subprocess.CompletedProcess(
                        [compiler_path, "-S", "-o", output_asm, src, "-O1"], 124, "", ""
                    ),
                )
            )
            return False

        if run_compiler_process.returncode != 0:
            self.result.cases_result[ResultType.RUN_COMPILER_FAILED].append(
                (src, run_compiler_process)
            )
            return False

        try:
            link_executable_process = link_executable(
                output_asm, target, output_exe, timeout=self.timeout
            )
        except subprocess.TimeoutExpired:
            print(Fore.RED + f"Test {src} link_executable timeout")
            self.result.cases_result[ResultType.LINK_EXECUTABLE_FAILED].append(
                (src, subprocess.CompletedProcess(["gcc link"], 124, "", ""))
            )
            return False

        if link_executable_process.returncode != 0:
            self.result.cases_result[ResultType.LINK_EXECUTABLE_FAILED].append(
                (src, link_executable_process)
            )
            return False

        try:
            res, process = run_executable(
                qemu_command + [output_exe], src, timeout=self.timeout
            )
        except subprocess.TimeoutExpired:
            print(Fore.RED + f"Test {src} run_executable timeout")
            self.result.cases_result[ResultType.RUN_EXECUTABLE_FAILED].append(
                (src, subprocess.CompletedProcess(["qemu"], 124, "", ""))
            )
            return False

        if not res:
            self.result.cases_result[ResultType.OUTPUT_MISMATCH].append((src, process))
            return False

        self.result.cases_result[ResultType.PASSED].append((src, process))
        return res

    def test_beta(self, path: str, suffix: str, tester):
        """
        test files with suffix in path with tester.
        for f in all files in path with suffix:
        tester(f)
        """
        print(f"Test files with suffix {suffix} in {path}")
        test_list = []
        for dirpath, dirnames, filenames in os.walk(path):
            for f in filenames:
                if f.endswith(suffix):
                    test_list.append(os.path.join(dirpath, f))
        cnt = 0
        failed_list = []
        test_list.sort()
        for src in test_list:
            cnt += 1
            print(Fore.YELLOW + f"Test {cnt}/{len(test_list)}: {src}")
            try:
                if tester(src) is not False:
                    print(Fore.GREEN + "Test passed")
                    continue
            except Exception as e:
                print(Fore.RED + f"Test failed: {e}")
            failed_list.append(src)

        return len(test_list), len(failed_list)


# test_instance = Test("riscv", "2023", "functional")
# test_instance.run()
# test_instance.run_single_case("2023/functional/75_max_flow.sy")
# test_instance.run_single_case("2023/functional/00_main.sy")
# test_instance.run_single_case("2023/functional/68_brainfk.sy")


# test_instance = Test("riscv", "2023", "hidden_functional")
# test_instance.run()
# test_instance.run_single_case("2023/hidden_functional/23_json.sy")

import time

if __name__ == "__main__":
    # timeout = 5
    test = Test("riscv", "2023", "functional")
    # test.run()
    test.run_single_case("2023/functional/00_main.sy")
    # time.sleep(2)
    # Test("riscv", "2023", "hidden_functional").run()
