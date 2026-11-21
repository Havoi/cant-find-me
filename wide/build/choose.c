#include <unistd.h>
#include <stdio.h>
extern int __real_main(int, char **);

int __wrap_main(int argc, char **argv) {
    __builtin_cpu_init();
    if (__builtin_cpu_supports("avx512ifma")) {
        const char *program = (__builtin_cpu_supports("avx512vbmi") && __builtin_cpu_supports("gfni")) ? "/vbmi" : "/ifma";
        execv(program, argv);
        perror(program);
        return 1;
    }
    return __real_main(argc, argv);
}
