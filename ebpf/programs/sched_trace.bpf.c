// SPDX-License-Identifier: GPL-2.0
/*
 * AACO GPU Scheduler Sampling eBPF Program
 * 
 * Traces GPU-related scheduler events to correlate
 * GPU workload with CPU scheduling decisions.
 */

#include <linux/bpf.h>
#include <linux/ptrace.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>

#define MAX_ENTRIES 10240
#define TASK_COMM_LEN 16

/* Event types */
#define EVENT_SCHED_SWITCH 1
#define EVENT_SCHED_WAKEUP 2
#define EVENT_GPU_SUBMIT   3

/* GPU scheduler event data */
struct sched_event {
    __u64 timestamp;
    __u32 cpu;
    __u32 pid;
    __u32 prev_pid;
    __u32 event_type;
    char comm[TASK_COMM_LEN];
    __u64 latency_ns;
};

/* Ring buffer for events */
struct {
    __uint(type, BPF_MAP_TYPE_RINGBUF);
    __uint(max_entries, 256 * 1024);
} events SEC(".maps");

/* Per-CPU timestamp tracking */
struct {
    __uint(type, BPF_MAP_TYPE_PERCPU_ARRAY);
    __type(key, __u32);
    __type(value, __u64);
    __uint(max_entries, 1);
} cpu_ts SEC(".maps");

/* PID filter (0 = trace all) */
struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __type(key, __u32);
    __type(value, __u32);
    __uint(max_entries, 1);
} pid_filter SEC(".maps");

/* Statistics */
struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __type(key, __u32);
    __type(value, __u64);
    __uint(max_entries, 8);
} stats SEC(".maps");

static __always_inline int should_trace(__u32 pid)
{
    __u32 key = 0;
    __u32 *filter_pid = bpf_map_lookup_elem(&pid_filter, &key);
    
    if (!filter_pid || *filter_pid == 0)
        return 1;  /* Trace all */
    
    return pid == *filter_pid;
}

SEC("tp/sched/sched_switch")
int trace_sched_switch(struct trace_event_raw_sched_switch *ctx)
{
    __u32 prev_pid = ctx->prev_pid;
    __u32 next_pid = ctx->next_pid;
    
    /* Filter by PID if set */
    if (!should_trace(prev_pid) && !should_trace(next_pid))
        return 0;
    
    struct sched_event *e;
    e = bpf_ringbuf_reserve(&events, sizeof(*e), 0);
    if (!e)
        return 0;
    
    e->timestamp = bpf_ktime_get_ns();
    e->cpu = bpf_get_smp_processor_id();
    e->pid = next_pid;
    e->prev_pid = prev_pid;
    e->event_type = EVENT_SCHED_SWITCH;
    bpf_get_current_comm(&e->comm, sizeof(e->comm));
    e->latency_ns = 0;
    
    /* Update switch count */
    __u32 stat_key = 0;
    __u64 *count = bpf_map_lookup_elem(&stats, &stat_key);
    if (count)
        __sync_fetch_and_add(count, 1);
    
    bpf_ringbuf_submit(e, 0);
    return 0;
}

SEC("tp/sched/sched_wakeup")
int trace_sched_wakeup(struct trace_event_raw_sched_wakeup *ctx)
{
    __u32 pid = ctx->pid;
    
    if (!should_trace(pid))
        return 0;
    
    struct sched_event *e;
    e = bpf_ringbuf_reserve(&events, sizeof(*e), 0);
    if (!e)
        return 0;
    
    e->timestamp = bpf_ktime_get_ns();
    e->cpu = bpf_get_smp_processor_id();
    e->pid = pid;
    e->prev_pid = 0;
    e->event_type = EVENT_SCHED_WAKEUP;
    bpf_probe_read_str(&e->comm, sizeof(e->comm), ctx->comm);
    e->latency_ns = 0;
    
    /* Update wakeup count */
    __u32 stat_key = 1;
    __u64 *count = bpf_map_lookup_elem(&stats, &stat_key);
    if (count)
        __sync_fetch_and_add(count, 1);
    
    bpf_ringbuf_submit(e, 0);
    return 0;
}

char LICENSE[] SEC("license") = "GPL";
