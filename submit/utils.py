import os
import shutil
import subprocess

import colorama
from colorama import Fore, Style


def isZero(x):
    return abs(x) < 1e-5

def removePathSuffix(filename: str):
    """
    basename("a/b/c.txt") => "a/b/c"
    """
    return os.path.splitext(filename)[0]


def overwritten_or_create_dir(path):
    if os.path.exists(path):
        print(f"Warning: {path} found, will be overwritten")
        shutil.rmtree(path)
        os.mkdir(path)
    else:
        print(f"{path} not found, will be created")
        os.mkdir(path)


def check_args(
    compiler_path, tests_path, output_asm_path, output_exe_path, output_c_path
):
    if not os.path.exists(compiler_path):
        print(f"Compiler not found: {compiler_path}")
        print("Please run: `python compile.py ./ compiler` first")
        return False
    if not os.path.exists(tests_path):
        print(f"Tests path not found: {tests_path}")
        return False
    overwritten_or_create_dir(output_asm_path)
    overwritten_or_create_dir(output_exe_path)
    overwritten_or_create_dir(output_c_path)
    return True


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
    output_file = removePathSuffix(src) + ".out"
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
                print(Fore.GREEN + f" {used:.6f}s")
            return max(1e-6, used)

    raise RuntimeError("No performance data")
