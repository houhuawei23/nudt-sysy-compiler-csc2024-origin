import sys
import subprocess

compiler_path = sys.argv[1]
test_path = sys.argv[2]

qemu_command = {
    "riscv": f"qemu-riscv64 -L /usr/riscv64-linux-gnu".split(),
    "arm": f"qemu-arm -L /usr/arm-linux-gnueabihf".split(),
}

# -w: suppress warnings
# -DNDEBUG: disable assertions
gcc_ref_command = "gcc -O3".split()
clang_ref_command = "clang -O3".split()

qemu_gcc_ref_command = {
    "riscv": "riscv64-linux-gnu-gcc -march=rv64gc -mabi=lp64".split(),
    "arm": "arm-linux-gnueabihf-gcc -march=armv7".split(),
}

targets = set(["riscv"])


def run_compiler(
    source_path,
    target,
    output="/dev/stdout",
    opt_level=0,
    log_level=1,
    emit_ir=False,
    input_file=None,
) -> subprocess.CompletedProcess:
    assert 0 <= opt_level <= 3, "Invalid optimization level"
    assert 0 <= log_level <= 3, "Invalid log level"
    command = [
        compiler_path,
        "-S",
        # "-t",
        # target,
        f"-O{opt_level}",
        f"-L{log_level}",
        "-o",
        output,
    ]
    if emit_ir:
        command.append("-i")
    
    command = command + ["-f", source_path]
    print(command)

    process = subprocess.run(command, capture_output=True, text=True)
    return process
out = run_compiler(test_path, "riscv", output=".vscode/test.s", opt_level=0, log_level=1, emit_ir=True)    


print(out.stderr)

print(out.stdout)

print(out.returncode)
