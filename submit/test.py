import os
import sys
import subprocess
import shutil

compiler_path = sys.argv[1]
tests_path = sys.argv[2]  #
output_asm_path = sys.argv[3]
output_exe_path = sys.argv[4]


def overwritten_or_create_dir(path):
    if os.path.exists(path):
        print(f"Warning: {path} found, will be overwritten")
        # os.rmdir(path)
        # os.removedirs(path)
        shutil.rmtree(path)
        os.mkdir(path)
    else:
        print(f"{path} not found, will be created")
        os.mkdir(path)


def check_args():
    if not os.path.exists(compiler_path):
        print(f"Compiler not found: {compiler_path}")
        print("Please run: `python compile.py ./ compiler` first")
        return False
    if not os.path.exists(tests_path):
        print(f"Tests path not found: {tests_path}")
        return False
    overwritten_or_create_dir(output_asm_path)
    overwritten_or_create_dir(output_exe_path)
    return True


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


def test_sysy_groups(groupname, target, test_func: callable):
    """
    test_func: callable(path: str, target: str) -> bool
    """
    testnum, failednum = test(
        f"SysY compiler functional {groupname}-{target}",
        os.path.join(tests_path, "2023/functional"),
        ".sy",
        lambda x: test_func(x, target),
    )
    print(f"Total {testnum} tests, {failednum} failed")


# test("SysY compiler functional", tests_path + "2021/functional", ".sy", )


def run_compiler(src, target, output, opt_level=0, debug_level=0, emit_ir=False):
    """
    ./compiler -S -o output src
    ./compiler -S -o output src -O1
    """
    command = [compiler_path, "-S", "-o", output, src]
    process = subprocess.run(command, capture_output=True, text=True)
    return process


def basename(filename: str):
    """
    basename("a/b/c.txt") => "a/b/c"
    """
    return os.path.splitext(filename)[0]


def link_executable(src: str, target: str, output: str, runtime=sysy_runtime):
    """
    riscv64-linux-gnu-gcc-12
    """
    command = qemu_gcc_ref_command + ["-o", output, runtime, src]
    process = subprocess.run(command, capture_output=True, text=True)
    if process.returncode != 0:
        raise RuntimeError(f"Linking failed: {process.stderr}")

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

    if output != standard_answer:
        print(" Output mismatch")
        print("output:", output[:10], end="")
        print("--------")
        print("stdans:", standard_answer[:10], end="")
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


def run_executable(command, src):
    input_file = basename(src) + ".in"
    if os.path.exists(input_file):
        with open(input_file, "r", encoding="utf-8") as f:
            out = subprocess.run(
                command, stdin=f, capture_output=True, text=True, timeout=1
            )
    else:
        out = subprocess.run(command, capture_output=True, text=True, timeout=1)

    # time_used = compare_and_parse_perf(src, out)
    # return time_used

    res = compare_output_with_standard_file(
        basename(src) + ".out", out.stdout, out.returncode
    )
    # if not res:
    #     print(out.stdout)
    return res

    if out.returncode == 0:
        print(out.stdout)
        return True
    print(out.stderr)
    # if timeout:
    if out.returncode == 124:
        print("Timeout")


# test_func: callable(src: str, target: str) -> bool
def sysy_compiler_qemu(src: str, target: str):
    filename = os.path.basename(src)
    raw_name = os.path.splitext(filename)[0]
    output_exe = os.path.join(output_exe_path, raw_name)
    output_asm = os.path.join(output_asm_path, raw_name + ".s")
    run_compiler(src, target, output_asm)
    link_executable(output_asm, target, output_exe)
    res = run_executable(qemu_command + [output_exe], src)
    return res


test_sysy_groups("qemu", "riscv", sysy_compiler_qemu)
