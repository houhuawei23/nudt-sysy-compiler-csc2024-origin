# python test.py compiler_path tests_path output_asm_path output_exe_path output_c_path
# python ./submit/test.py ./compiler ./test/ ./.tmp/asm ./.tmp/exe ./.tmp/c


import os
import sys
import subprocess
import time
from datetime import datetime


from utils import (
    removePathSuffix,
    compare_output_with_standard_file,
    compare_and_parse_perf,
)
from TestResult import TestResult, ResultType, colorMap

import colorama
from colorama import Fore, Style

# Initializes colorama and autoresets color
colorama.init(autoreset=True)

from concurrent.futures import ThreadPoolExecutor, as_completed

stack_size = 128 << 20  # 128M

qemu_command = f"qemu-riscv64 -L /usr/riscv64-linux-gnu/ -cpu rv64,zba=true,zbb=true -s {stack_size} -D /dev/stderr".split()

gcc_ref_command = "gcc -x c++ -O3 -DNDEBUG -march=native -fno-tree-vectorize -s -funroll-loops -ffp-contract=on -w ".split()

clang_ref_command = "clang -Qn -O3 -DNDEBUG -emit-llvm -fno-slp-vectorize -fno-vectorize -mllvm -vectorize-loops=false -S -ffp-contract=on -w ".split()

qemu_gcc_ref_command = "riscv64-linux-gnu-gcc-12 -O2 -DNDEBUG -march=rv64gc -mabi=lp64d -mcmodel=medlow -ffp-contract=on -w ".split()

qemu_gpp_ref_command = "riscv64-linux-gnu-g++-12 -O2 -DNDEBUG -march=rv64gc -mabi=lp64d -mcmodel=medlow -ffp-contract=on -w ".split()


def test(path: str, suffix: str, tester):
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


def run_compiler(
    compiler_path,
    src,
    target,
    output,
    opt_level=0,
    debug_level=0,
    emit_ir=False,
    timeout=1,
):
    """
    ./compiler -S -o output src
    ./compiler -S -o output src -O1
    """
    command = [compiler_path, "-S", "-o", output, src, f"-O{opt_level}"]
    # print(*command, sep=" ")
    process = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    
    # os.sched_setaffinity(process.pid, {core})
    return process


def run_riscv_gcc(src, target, output, opt_level=0, debug_level=0, timeout=1):
    """
    riscv64-linux-gnu-gcc-12 -S -o output src
    """
    command = qemu_gpp_ref_command + ["-S", "-o", output, src, f"-O{opt_level}"]
    # print(*command, sep=" ")
    print(*command, sep=" ")
    process = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    return process


def link_executable(src: str, target: str, output: str, runtime, timeout=1):
    """
    riscv64-linux-gnu-gcc-12
    """
    command = qemu_gcc_ref_command + ["-o", output, runtime, src]
    # print(*command, sep=" ")
    process = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    return process


def link_ricvgpp_executable(src: str, target: str, output: str, timeout=1):
    """
    riscv64-linux-gnu-gcc-12
    """
    command = qemu_gpp_ref_command + ["-o", output, src]
    # print(*command, sep=" ")
    process = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    return process


def run_executable(command, src, timeout=1):
    input_file = removePathSuffix(src) + ".in"
    # print(*command, sep=" ")
    if os.path.exists(input_file):
        with open(input_file, "r", encoding="utf-8") as f:
            out = subprocess.run(
                command, stdin=f, capture_output=True, text=True, timeout=timeout
            )
    else:
        out = subprocess.run(command, capture_output=True, text=True, timeout=timeout)

    output_file = removePathSuffix(src) + ".out"
    res = compare_output_with_standard_file(output_file, out.stdout, out.returncode)

    return res, out


def test(path: str, suffix: str, tester):
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


class Test:

    def __init__(
        self,
        compiler_path="./compiler",
        tests_path="./test",
        output_asm_path="./.tmp/asm",
        output_exe_path="./.tmp/exe",
        output_c_path="./.tmp/c",
        runtime="./test/sysy/sylib.c",
        sysy_link_for_riscv_gpp="./test/link/link.c",
    ):
        self.compiler_path = compiler_path
        self.tests_path = tests_path
        self.output_asm_path = output_asm_path
        self.output_exe_path = output_exe_path
        self.output_c_path = output_c_path
        self.runtime = runtime
        self.sysy_link_for_riscv_gpp = sysy_link_for_riscv_gpp

    def set(self, target, year, timeout=5):
        self.target = target
        self.year = year
        self.timeout = timeout
        self.result = TestResult(f"SysY compiler {year}")

    def run_single_test(self, src: str, tester):
        try:
            if tester(src) is not False:
                print(Fore.GREEN + "Test passed")
                return src, True
        except Exception as e:
            print(Fore.RED + f"Test failed: {e}")
        return src, False

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
        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(self.run_single_test, src, tester): src
                for src in test_list
            }
            for future in as_completed(futures):
                cnt += 1
                src = futures[future]
                print(Fore.YELLOW + f"Test {cnt}/{len(test_list)}: {src}")
                result = future.result()
                if not result[1]:
                    failed_list.append(src)

        return len(test_list), len(failed_list)

    def run(self, test_kind: str):
        print(Fore.RED + f"Testing {self.year} {test_kind}...")
        year_kind_path = os.path.join(self.tests_path, self.year, test_kind)
        testnum, failednum = self.test_beta(
            year_kind_path,
            ".sy",
            lambda x: self.sysy_compiler_qemu(x, self.target),
        )
        self.result.print_result_overview()
        dt_string = datetime.now().strftime("%Y_%m_%d_%H:%M")
        self.result.save_result(f"./.{self.year}_{test_kind}_{dt_string}.md")

    def run_perf(self, test_kind: str):
        print(Fore.RED + f"Testing {self.year} {test_kind}...")
        year_kind_path = os.path.join(self.tests_path, self.year, test_kind)
        testnum, failednum = self.test_beta(
            year_kind_path,
            ".sy",
            lambda x: self.sysy_qemu_perf(x, self.target),
        )
        self.result.print_perf_overview()
        dt_string = datetime.now().strftime("%Y_%m_%d_%H:%M")
        self.result.save_perf_result(f"./.{self.year}_{test_kind}_{dt_string}.md")

    def run_single_case(self, test_kind: str, filename: str):
        """
        test.run_single_case("functional", "04_arr_defn3.sy")
        """
        test_case_path = os.path.join(self.tests_path, self.year, test_kind, filename)
        self.sysy_compiler_qemu(test_case_path, self.target)
        self.sysy_gcc_qemu(test_case_path, self.target)
        # self.result.print_result_overview()
        for type in ResultType:
            if len(self.result.cases_result[type]) > 0:
                self.result.print_result(type)
        pass

    def sysy_compiler_qemu(self, src: str, target: str = "riscv"):
        if os.path.exists(src) is False:
            print(f"Test file not found: {src}")
            return False
        filename = os.path.basename(src)
        raw_name = os.path.splitext(filename)[0]  # abc.sy -> abc
        output_exe = os.path.join(self.output_exe_path, raw_name)
        output_asm = os.path.join(self.output_asm_path, raw_name + ".s")

        try:
            run_compiler_process = run_compiler(
                self.compiler_path, src, target, output_asm, timeout=self.timeout
            )
        except subprocess.TimeoutExpired:
            print(Fore.RED + f"Test {src} run_compiler timeout")
            self.result.cases_result[ResultType.RUN_COMPILER_FAILED].append(
                (
                    src,
                    subprocess.CompletedProcess(
                        [self.compiler_path, "-S", "-o", output_asm, src, "-O1"],
                        124,
                        "",
                        "",
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
                output_asm, target, output_exe, self.runtime, timeout=self.timeout
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

        time_used = compare_and_parse_perf(src, process)
        if src in self.result.qemu_run_time:
            self.result.qemu_run_time[src] = (
                time_used,
                self.result.qemu_run_time[src][1],
            )
        else:
            self.result.qemu_run_time[src] = (time_used, 0)

        self.result.cases_result[ResultType.PASSED].append((src, process))
        return res

    def sysy_gcc_qemu(self, src_path: str, target: str = "riscv"):
        if os.path.exists(src_path) is False:
            print(f"Test file not found: {src_path}")
            return False
        # src_cpath = removePathSuffix(src_path) + ".c"
        src_filename = os.path.basename(src_path)  # path/to/abc.sy -> abc.sy
        src_cpath = os.path.join(
            self.output_c_path, removePathSuffix(os.path.basename(src_path)) + ".c"
        )
        src_raw_name = os.path.splitext(src_filename)[0]  # abc.sy -> abc
        # path/to/output/abc
        output_exe = os.path.join(self.output_exe_path, src_raw_name + "_gcc")
        # path/to/output/abc.s
        output_asm = os.path.join(self.output_asm_path, src_raw_name + "_gcc" ".s")

        # prepare src_cpath
        with open(self.sysy_link_for_riscv_gpp, "r", encoding="utf-8") as f:
            link_code = f.read()
        with open(src_path, "r", encoding="utf-8") as f:
            sy_code = f.read()
        with open(src_cpath, "w", encoding="utf-8") as f:
            f.write(link_code + "\n\n" + sy_code)

        process = run_riscv_gcc(
            src_cpath, target, output_asm, opt_level=3, timeout=self.timeout
        )

        process = link_ricvgpp_executable(
            src_cpath, target, output_exe, timeout=self.timeout
        )

        res, process = run_executable(
            qemu_command + [output_exe], src_path, timeout=self.timeout
        )
        time_used = compare_and_parse_perf(src_path, process)
        if src_path in self.result.qemu_run_time:
            self.result.qemu_run_time[src_path] = (
                self.result.qemu_run_time[src_path][0],
                time_used,
            )
        else:
            self.result.qemu_run_time[src_path] = (0, time_used)

        return res

    def sysy_qemu_perf(self, src_path: str, target: str = "riscv"):
        if os.path.exists(src_path) is False:
            print(f"Test file not found: {src_path}")
            return False

        compiler_res = self.sysy_compiler_qemu(src_path, target)
        gcc_res = self.sysy_gcc_qemu(src_path, target)
        if not (compiler_res and gcc_res):
            raise Exception("Compiler or gcc failed")
        return compiler_res and gcc_res
