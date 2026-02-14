// SPDX-License-Identifier: GPL-2.0
/*
 * AACO GPU Memory Tracking eBPF Program
 * 
 * Tracks GPU memory allocation patterns via kprobes
 * on AMD GPU driver functions.
 */

#include <linux/bpf.h>
#include <linux/ptrace.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include <bpf/bpf_core_read.h>

#define MAX_STACK_DEPTH 16

/* Memory event types */
#define MEM_ALLOC   1
#define MEM_FREE    2
#define MEM_MAP     3
#define MEM_UNMAP   4

/* Memory event structure */
struct mem_event {
    __u64 timestamp;
    __u32 pid;
    __u32 event_type;
    __u64 size;
    __u64 addr;
    __u64 gpu_addr;
    __s32 gpu_id;
    __u32 flags;
};

/* Allocation tracking */
struct alloc_info {
    __u64 size;
    __u64 timestamp;
    __u32 pid;
};

/* Ring buffer for memory events */
struct {
    __uint(type, BPF_MAP_TYPE_RINGBUF);
    __uint(max_entries, 256 * 1024);
} mem_events SEC(".maps");

/* Track outstanding allocations */
struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __type(key, __u64);          /* GPU address */
    __type(value, struct alloc_info);
    __uint(max_entries, 65536);
} allocations SEC(".maps");

/* Per-GPU memory statistics */
struct {
    __uint(type, BPF_MAP_TYPE_PERCPU_ARRAY);
    __type(key, __u32);
    __type(value, __u64);
    __uint(max_entries, 16);     /* Stats per GPU */
} gpu_mem_stats SEC(".maps");

/* Stack traces for allocations */
struct {
    __uint(type, BPF_MAP_TYPE_STACK_TRACE);
    __uint(key_size, sizeof(__u32));
    __uint(value_size, MAX_STACK_DEPTH * sizeof(__u64));
    __uint(max_entries, 4096);
} stack_traces SEC(".maps");

/*
 * Kprobe on amdgpu_bo_create
 * Traces VRAM allocations
 */
SEC("kprobe/amdgpu_bo_create")
int trace_gpu_alloc(struct pt_regs *ctx)
{
    struct mem_event *e;
    
    e = bpf_ringbuf_reserve(&mem_events, sizeof(*e), 0);
    if (!e)
        return 0;
    
    e->timestamp = bpf_ktime_get_ns();
    e->pid = bpf_get_current_pid_tgid() >> 32;
    e->event_type = MEM_ALLOC;
    
    /* Read size from function argument */
    /* Note: Actual arg positions depend on kernel version */
    e->size = PT_REGS_PARM2(ctx);
    e->addr = 0;
    e->gpu_addr = 0;
    e->gpu_id = 0;
    e->flags = PT_REGS_PARM3(ctx);
    
    bpf_ringbuf_submit(e, 0);
    return 0;
}

/*
 * Kretprobe on amdgpu_bo_create
 * Captures return value (allocated address)
 */
SEC("kretprobe/amdgpu_bo_create")
int trace_gpu_alloc_ret(struct pt_regs *ctx)
{
    int ret = PT_REGS_RC(ctx);
    
    /* Update stats on successful allocation */
    if (ret == 0) {
        __u32 key = 0;
        __u64 *total = bpf_map_lookup_elem(&gpu_mem_stats, &key);
        if (total)
            __sync_fetch_and_add(total, 1);
    }
    
    return 0;
}

/*
 * Kprobe on amdgpu_bo_release
 * Traces VRAM frees
 */
SEC("kprobe/amdgpu_bo_release")
int trace_gpu_free(struct pt_regs *ctx)
{
    struct mem_event *e;
    
    e = bpf_ringbuf_reserve(&mem_events, sizeof(*e), 0);
    if (!e)
        return 0;
    
    e->timestamp = bpf_ktime_get_ns();
    e->pid = bpf_get_current_pid_tgid() >> 32;
    e->event_type = MEM_FREE;
    e->size = 0;
    e->addr = PT_REGS_PARM1(ctx);
    e->gpu_addr = 0;
    e->gpu_id = 0;
    e->flags = 0;
    
    bpf_ringbuf_submit(e, 0);
    return 0;
}

/*
 * Kprobe on amdgpu_vm_bo_map
 * Traces GPU virtual memory mappings
 */
SEC("kprobe/amdgpu_vm_bo_map") 
int trace_gpu_map(struct pt_regs *ctx)
{
    struct mem_event *e;
    
    e = bpf_ringbuf_reserve(&mem_events, sizeof(*e), 0);
    if (!e)
        return 0;
    
    e->timestamp = bpf_ktime_get_ns();
    e->pid = bpf_get_current_pid_tgid() >> 32;
    e->event_type = MEM_MAP;
    e->size = 0;
    e->addr = 0;
    e->gpu_addr = PT_REGS_PARM3(ctx);
    e->gpu_id = 0;
    e->flags = PT_REGS_PARM4(ctx);
    
    bpf_ringbuf_submit(e, 0);
    return 0;
}

char LICENSE[] SEC("license") = "GPL";
