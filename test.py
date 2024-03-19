import os
import argparse
import subprocess

from colorama import Fore, Style


from colorama import init
init(autoreset=True)

def print_err(stderr):
    print(Fore.RED + stderr)

def print_succ():
    print(Fore.GREEN + "Success")


PASS_CNT = 0
ERROR_CNT = 0
SKIP_CNT = 0
error_file_list = []
test_dir = "test/steps"
test_file = "test.c"

output_dir = "test/.out"
# result_file = "test/.out/result.txt"

sysylib_funcs = [
    "getint", "getch", "getarray", "getfloat", "getfarray", 
    "putint",  "putch", "putarray", "putfloat", "putfarray", "putf", 
    "starttime", "stoptime"]
# Function to print usage information
def usage():
    print("Usage: python script.py [-t <test_dir>] [-o <output_dir>] [-s <file>] [-r <result_file>] [-h]")
    print("Options:")
    print("  -t <test_dir>     Specify the directory containing test files (default: test/steps/)")
    print("  -o <output_dir>   Specify the output directory (default: test/.out/)")
    # print("  -r <result_file>  Specify the file to store the test results (default: test/.out/result.txt)")
    print("  -h                Print this help message")

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("-t", "--target", help="Specify the target dir or file")
parser.add_argument("-o", "--output_dir", help="Specify the output directory", default=output_dir)
# parser.add_argument("-r", "--result_file", help="Specify the file to store the test results", default=result_file)
args = parser.parse_args()

output_dir = args.output_dir
# result_file = args.result_file

genir_file = output_dir + "/gen.ll"
llvmir_file = output_dir + "/llvm.ll"

is_file = False

if os.path.isfile(args.target):
    test_file_path = args.target
    test_file = os.path.basename(test_file_path)
    is_file = True
elif os.path.isdir(args.target):
    test_dir = args.target
else:
    print("Error: Invalid target!")
    exit(1)

# Ensure output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


import re

def check_keywords_in_file(file_path, keywords):
    # Create a regular expression pattern that matches any of the keywords
    pattern = re.compile('|'.join(keywords))
    
    with open(file_path, 'r') as file:
        # Read the entire file as a single string
        file_contents = file.read()
        # Use findall to find all occurrences of the pattern in the file contents
        matches = pattern.findall(file_contents)
        
        # If there are any matches, return True
        if matches:
            return True
    return False


def run_test_file(file_path):
    global PASS_CNT, SKIP_CNT, ERROR_CNT
    global error_file_list
    is_libfunc = check_keywords_in_file(file_path, sysylib_funcs)
    if is_libfunc:
        print(Fore.MAGENTA + "[SKIP] ", end="")
        print(Style.DIM + file_path)
        SKIP_CNT = SKIP_CNT + 1
        return 

    # run main test.sy
    run_res = subprocess.run(["./main", file_path], capture_output=True, text=True)
    if run_res.returncode != 0:
        print(Fore.RED + "[ERROR] main: ", end="")
        print(Fore.RED + file_path)
        print(Style.BRIGHT + "    "  + f"return code: {run_res.returncode}") # Fore.MAGENTA
        ERROR_CNT = ERROR_CNT + 1
        error_file_list.append(file_path)
        return 
    if run_res.stderr:
        print(Fore.RED + "[ERROR] main: ", end="")
        print(Fore.RED + file_path)

        print(Style.BRIGHT + "    " + run_res.stderr.strip()) # Fore.MAGENTA
        ERROR_CNT = ERROR_CNT + 1
        error_file_list.append(file_path)
        return 
    # write gen.ll
    with open(genir_file, "w") as f:
        f.write(run_res.stdout)
    
    # run lli
    lli_res = subprocess.run(["lli", genir_file], capture_output=True, text=True)
    if lli_res.stderr:
        print(Fore.RED + "[ERROR] lli: ", end="")
        print(Fore.RED + file_path)
        print(Style.BRIGHT + lli_res.stderr.strip()) # Fore.MAGENTA
        ERROR_CNT = ERROR_CNT + 1
        error_file_list.append(file_path)
        return 
    
    print(Fore.GREEN + "[PASS] ", end="")
    print(Style.DIM + file_path)
    PASS_CNT = PASS_CNT + 1
    # print(lli_res.stdout)

    # if test_file[-3:] == '.sy':
    #     new_test_file_path = "test/.out/" + 'tmp.c'
    #     subprocess.run(["cp", test_file_path, new_test_file_path])
    # else:
    #     new_test_file_path = test_file_path

if __name__ == "__main__":
    if is_file:
        run_test_file(test_file_path)
    else:
        files = os.listdir(test_dir)
        files.sort()
        # print(files)
        for file in files:
            if file.endswith((".c", ".sy")):
                path = test_dir + "/" + file
                run_test_file(path)
    TOTAL_CNT = PASS_CNT + ERROR_CNT + SKIP_CNT
    # aligned
    # print(f"Total: {TOTAL_CNT:>5}")

    print(Fore.RED + "[ERROR]")
    for f in error_file_list:
        print(f"{f}")
    
    print(Fore.YELLOW + "[INFO]")   
    print(f"Total: {TOTAL_CNT:>5}")
    print(f"Pass:  {PASS_CNT:>5}")
    print(f"Error: {ERROR_CNT:>5}")
    print(f"Skip:  {SKIP_CNT:>5}")