
from enum import Enum


import colorama
from colorama import Fore, Style

# Initializes colorama and autoresets color
colorama.init(autoreset=True)  



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
        for src, process in self.cases_result[type]:
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
