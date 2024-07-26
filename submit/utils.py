import os
import sys
import subprocess
import shutil


def overwritten_or_create_dir(path):
    if os.path.exists(path):
        print(f"Warning: {path} found, will be overwritten")
        shutil.rmtree(path)
        os.mkdir(path)
    else:
        print(f"{path} not found, will be created")
        os.mkdir(path)


def check_args(compiler_path, tests_path, output_asm_path, output_exe_path):
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
