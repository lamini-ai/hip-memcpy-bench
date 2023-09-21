#!/bin/bash

# Safely execute this bash script
# e exit on first failure
# x all executed commands are printed to the terminal
# u unset variables are errors
# a export all variables to the environment
# E any trap on ERR is inherited by shell functions
# -o pipefail | produces a failure code if any stage fails
set -Eeuoxa pipefail

# Get the directory of this script
LOCAL_DIRECTORY="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Create the build directory if it doesn't exist
mkdir -p $LOCAL_DIRECTORY/../build

# Use LLVM to compile the C++ code
clang++ -std=c++20 -D__HIP_PLATFORM_AMD__ -I /opt/rocm/include -I/usr/include/c++/11 -I/usr/include/x86_64-linux-gnu/c++/11 -L /usr/lib/gcc/x86_64-linux-gnu/11 -L /opt/rocm/lib  -o $LOCAL_DIRECTORY/../build/rocblas-bench $LOCAL_DIRECTORY/../src/main.cpp -l rocblas -l amdhip64



