import sys
import os
import subprocess

compiler_path = sys.argv[1]
test_path = sys.argv[2]
out_path = sys.argv[3]

# python gen.py ./compiler ./test/2021/functional ./test/2021/functional/asm

def run_compiler(
    source_path,
    output="/dev/stdout",
    opt_level=0,
    log_level=0,
    emit_ir=False,
    input_file=None,
    timeout=3
) -> subprocess.CompletedProcess:
    assert 0 <= opt_level <= 3, "Invalid optimization level"
    assert 0 <= log_level <= 3, "Invalid log level"
    command = [
        compiler_path,
        "-S",
        f"-O{opt_level}",
        f"-L{log_level}",
        "-o",
        output,
    ]
    if emit_ir:
        command.append("-i")
    
    command = command + ["-f", source_path]
    print(" ".join(command))
    try:
        process = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    # except all exceptions
    except Exception as e:
        # timeout exeception
        if isinstance(e, subprocess.TimeoutExpired):
            print(f"Timeout: {timeout}s")
        else:
            print(f"Failed to process {source_path}. Error: {e}")
        return None
    return process


def generate_asm(test_dir, out_dir):
    for root, dirs, files in os.walk(test_dir):
        files.sort()
        for file in files:
            file_path = os.path.join(root, file)
            outfile_path = os.path.join(out_dir, file.replace(".sy", ".s"))
            if not file_path.endswith(".sy"):
                continue
            result = run_compiler(file_path, outfile_path)


generate_asm(test_path, out_path)