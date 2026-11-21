#!/bin/sh
set -eu
arch_flags="-march=x86-64-v3 -mtune=generic"
output=${1:-shopping-sessions}
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
compile() {
  ${CXX:-g++} -std=c++20 -O3 -DNDEBUG $arch_flags -static-libstdc++ -static-libgcc -Wa,-mbranches-within-32B-boundaries -flto=auto -pthread "$@" \
    main.cpp links.cpp ingest.cpp aggregate.cpp output.cpp -l:liblz4.so.1 -o "$output"
}
# A baseline amd64 builder can cross-compile the v3 executable. Run the
# instrumented training binary only on hosts that can execute it.
echo 'int main(void) { return !__builtin_cpu_supports("x86-64-v3"); }' >"$work/cpu.c"
cc "$work/cpu.c" -o "$work/cpu"
if ! "$work/cpu"; then compile; exit 0; fi
# The normal executable exits immediately after closing Parquet. Flush GCC's
# counters at precisely that point in the temporary instrumented executable.
cat >"$work/dump.c" <<'C'
extern void __gcov_dump(void);
extern void __real__exit(int);
void __wrap__exit(int status) { __gcov_dump(); __real__exit(status); }
C
cc -c "$work/dump.c" -o "$work/dump.o"
compile "-fprofile-generate=$work/counters" "$work/dump.o" -Wl,--wrap=_exit
python3 training.py "$work/input"
NCPU=4 "$output" "$work/input" "$work/output"
compile "-fprofile-use=$work/counters" -fprofile-correction -Werror=missing-profile
