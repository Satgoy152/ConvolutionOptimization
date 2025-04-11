#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <cstdint>
#include <errno.h>
#include <linux/audit.h>
#include <linux/filter.h>
#include <linux/seccomp.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/prctl.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <vector>

static __attribute__((constructor)) void setup_seccomp_filter(void) {
    if (getenv("DISABLE_SANDBOX")) return;
    std::vector<struct sock_filter> filter = {
        // Load architecture
        BPF_STMT(BPF_LD | BPF_W | BPF_ABS, (offsetof(struct seccomp_data, arch))),
        // Check architecture (AUDIT_ARCH_X86_64)
        BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, AUDIT_ARCH_X86_64, 1, 0),
        // Kill if not AMD64
        BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_KILL),
        // Load syscall number
        BPF_STMT(BPF_LD | BPF_W | BPF_ABS, (offsetof(struct seccomp_data, nr)))};

    // For each allowed syscall, create a check
    for (uint32_t allowed_syscall :
         {SYS_brk,        SYS_clock_gettime, SYS_close,          SYS_exit,
          SYS_exit_group, SYS_fcntl,         SYS_fstat,          SYS_futex,
          SYS_getpid,     SYS_ioctl,         SYS_lseek,          SYS_madvise,
          SYS_mmap,       SYS_mprotect,      SYS_munmap,         SYS_poll,
          SYS_read,       SYS_rt_sigaction,  SYS_rt_sigprocmask, SYS_write}) {
        // Jump to allow if syscall matches
        filter.push_back(BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, allowed_syscall, 0, 1));
        filter.push_back(BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW));
    }

    // Default action: return errno
    filter.push_back(
        BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ERRNO | (EPERM & SECCOMP_RET_DATA)));

    // Prepare the seccomp filter program
    struct sock_fprog prog = {
        .len = static_cast<unsigned short>(filter.size()),
        .filter = filter.data(),
    };

    // Enable seccomp filtering
    if (prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0)) {
        perror("prctl(PR_SET_NO_NEW_PRIVS)");
        exit(1);
    }

    // Set thread sync
    if (syscall(SYS_seccomp, SECCOMP_SET_MODE_FILTER, SECCOMP_FILTER_FLAG_TSYNC, &prog)) {
        perror("prctl(PR_SET_SECCOMP)");
        exit(1);
    }
}
