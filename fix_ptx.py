#!/usr/bin/env python3
from fileinput import input


def fix(ptx_lines):
    ptx_lines = iter(ptx_lines)
    # Unnecessary after llvm 11 due to my patch being accepted
    # for line in ptx_lines:
    #     if line.startswith("//") or line == "\n":
    #         yield line
    #     elif line.startswith(".version"):
    #         yield line
    #         break
    for line in ptx_lines:
        yield line.replace("div.rn.f32", "div.approx.ftz.f32")


# file is moved to a backup file and standard output is directed to the file
with input("montey/kernel.ptx", inplace=True) as f:
    for line in fix(f):
        print(line, end="")
