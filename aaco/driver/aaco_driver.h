/*
 * AACO-SIGMA Linux Kernel Driver Header
 * 
 * Provides kernel-level primitives for deterministic measurement:
 * - High-precision TSC timestamps
 * - Memory barriers
 * - CPU affinity control
 * - Preemption control
 * - IRQ statistics
 * 
 * Copyright 2024 AMD AI Compute Observatory
 * SPDX-License-Identifier: GPL-2.0-only
 */

#ifndef _AACO_DRIVER_H
#define _AACO_DRIVER_H

#include <linux/ioctl.h>
#include <linux/types.h>

/* Driver identification */
#define AACO_DRIVER_NAME    "aaco_driver"
#define AACO_DEVICE_NAME    "aaco"
#define AACO_CLASS_NAME     "aaco"

/* Version: 1.0.0 = 0x010000 */
#define AACO_VERSION_MAJOR  1
#define AACO_VERSION_MINOR  0
#define AACO_VERSION_PATCH  0
#define AACO_VERSION        ((AACO_VERSION_MAJOR << 16) | \
                             (AACO_VERSION_MINOR << 8) | \
                             AACO_VERSION_PATCH)

/* IOCTL magic number */
#define AACO_IOC_MAGIC      'A'

/* =============================================================================
 * Data Structures
 * ============================================================================= */

/* TSC timestamp with metadata */
struct aaco_tsc_reading {
    __u64 tsc_value;
    __u64 timestamp_ns;  /* fallback: ktime_get_ns() */
    __u32 cpu_id;
    __u32 flags;
};

/* CPU affinity mask (up to 64 CPUs) */
struct aaco_cpu_mask {
    __u64 mask;
    __u32 cpu_count;
    __u32 reserved;
};

/* IRQ statistics */
struct aaco_irq_stats {
    __u64 total_irqs;
    __u64 timer_irqs;
    __u64 ipi_irqs;
    __u64 other_irqs;
};

/* Context switch statistics */
struct aaco_ctx_stats {
    __u64 voluntary_switches;
    __u64 involuntary_switches;
    __u64 total_switches;
    __u64 reserved;
};

/* Measurement session state */
struct aaco_measurement_state {
    __u64 start_tsc;
    __u64 end_tsc;
    __u64 start_ns;
    __u64 end_ns;
    struct aaco_irq_stats start_irqs;
    struct aaco_irq_stats end_irqs;
    struct aaco_ctx_stats start_ctx;
    struct aaco_ctx_stats end_ctx;
    __u32 active;
    __u32 preempt_disabled;
};

/* Memory region for pinning */
struct aaco_memory_region {
    __u64 address;
    __u64 size;
    __u32 flags;
    __u32 reserved;
};

/* Flags for memory pinning */
#define AACO_MEM_FLAG_READ   (1 << 0)
#define AACO_MEM_FLAG_WRITE  (1 << 1)
#define AACO_MEM_FLAG_EXEC   (1 << 2)
#define AACO_MEM_FLAG_LOCK   (1 << 3)

/* Noise event record */
struct aaco_noise_event {
    __u64 timestamp_ns;
    __u32 type;          /* enum aaco_noise_type */
    __u32 severity;
    __u64 data[4];       /* Type-specific data */
};

enum aaco_noise_type {
    AACO_NOISE_IRQ = 1,
    AACO_NOISE_CONTEXT_SWITCH = 2,
    AACO_NOISE_PAGE_FAULT = 3,
    AACO_NOISE_PREEMPTION = 4,
    AACO_NOISE_MIGRATION = 5,
    AACO_NOISE_THROTTLE = 6,
};

/* =============================================================================
 * IOCTL Commands
 * ============================================================================= */

/* IOCTL command numbers */
enum aaco_ioctl_cmd {
    AACO_IOC_GET_VERSION = 0,
    AACO_IOC_GET_TSC_FREQ = 1,
    AACO_IOC_READ_TSC = 2,
    AACO_IOC_SET_CPU_AFFINITY = 3,
    AACO_IOC_GET_CPU_AFFINITY = 4,
    AACO_IOC_MEMORY_BARRIER = 5,
    AACO_IOC_DISABLE_PREEMPTION = 6,
    AACO_IOC_ENABLE_PREEMPTION = 7,
    AACO_IOC_GET_NOISE_COUNTER = 8,
    AACO_IOC_RESET_NOISE_COUNTER = 9,
    AACO_IOC_START_MEASUREMENT = 10,
    AACO_IOC_STOP_MEASUREMENT = 11,
    AACO_IOC_GET_IRQ_STATS = 12,
    AACO_IOC_GET_CONTEXT_SWITCHES = 13,
    AACO_IOC_PIN_MEMORY = 14,
    AACO_IOC_UNPIN_MEMORY = 15,
    AACO_IOC_GET_MEASUREMENT_STATE = 16,
};

/* IOCTL definitions */
#define AACO_GET_VERSION        _IOR(AACO_IOC_MAGIC, AACO_IOC_GET_VERSION, __u32)
#define AACO_GET_TSC_FREQ       _IOR(AACO_IOC_MAGIC, AACO_IOC_GET_TSC_FREQ, __u64)
#define AACO_READ_TSC           _IOR(AACO_IOC_MAGIC, AACO_IOC_READ_TSC, struct aaco_tsc_reading)
#define AACO_SET_CPU_AFFINITY   _IOW(AACO_IOC_MAGIC, AACO_IOC_SET_CPU_AFFINITY, struct aaco_cpu_mask)
#define AACO_GET_CPU_AFFINITY   _IOR(AACO_IOC_MAGIC, AACO_IOC_GET_CPU_AFFINITY, struct aaco_cpu_mask)
#define AACO_MEMORY_BARRIER     _IO(AACO_IOC_MAGIC, AACO_IOC_MEMORY_BARRIER)
#define AACO_DISABLE_PREEMPTION _IO(AACO_IOC_MAGIC, AACO_IOC_DISABLE_PREEMPTION)
#define AACO_ENABLE_PREEMPTION  _IO(AACO_IOC_MAGIC, AACO_IOC_ENABLE_PREEMPTION)
#define AACO_GET_IRQ_STATS      _IOR(AACO_IOC_MAGIC, AACO_IOC_GET_IRQ_STATS, struct aaco_irq_stats)
#define AACO_GET_CTX_SWITCHES   _IOR(AACO_IOC_MAGIC, AACO_IOC_GET_CONTEXT_SWITCHES, struct aaco_ctx_stats)
#define AACO_START_MEASUREMENT  _IO(AACO_IOC_MAGIC, AACO_IOC_START_MEASUREMENT)
#define AACO_STOP_MEASUREMENT   _IOR(AACO_IOC_MAGIC, AACO_IOC_STOP_MEASUREMENT, struct aaco_measurement_state)
#define AACO_PIN_MEMORY         _IOW(AACO_IOC_MAGIC, AACO_IOC_PIN_MEMORY, struct aaco_memory_region)
#define AACO_UNPIN_MEMORY       _IOW(AACO_IOC_MAGIC, AACO_IOC_UNPIN_MEMORY, struct aaco_memory_region)
#define AACO_GET_MEASUREMENT    _IOR(AACO_IOC_MAGIC, AACO_IOC_GET_MEASUREMENT_STATE, struct aaco_measurement_state)

/* =============================================================================
 * Kernel-internal Definitions (ifdef __KERNEL__)
 * ============================================================================= */

#ifdef __KERNEL__

#include <linux/fs.h>
#include <linux/cdev.h>
#include <linux/device.h>

/* Maximum concurrent measurement sessions */
#define AACO_MAX_SESSIONS   64

/* Per-file state */
struct aaco_file_state {
    struct aaco_measurement_state measurement;
    struct aaco_cpu_mask saved_affinity;
    bool affinity_modified;
    bool preempt_disabled;
    spinlock_t lock;
};

/* Driver state */
struct aaco_driver_state {
    dev_t devno;
    struct cdev cdev;
    struct class *class;
    struct device *device;
    
    /* TSC frequency (cached) */
    u64 tsc_freq_hz;
    
    /* Global statistics */
    atomic64_t total_measurements;
    atomic64_t total_noise_events;
    
    /* Per-CPU noise tracking */
    struct aaco_noise_event __percpu *last_noise;
};

/* Function prototypes */
int aaco_driver_init(void);
void aaco_driver_exit(void);
long aaco_ioctl(struct file *file, unsigned int cmd, unsigned long arg);
int aaco_open(struct inode *inode, struct file *file);
int aaco_release(struct inode *inode, struct file *file);

/* TSC helpers */
static inline u64 aaco_read_tsc(void)
{
    u32 lo, hi;
    asm volatile("rdtscp" : "=a"(lo), "=d"(hi) :: "ecx");
    return ((u64)hi << 32) | lo;
}

static inline u64 aaco_read_tsc_serialized(void)
{
    u32 lo, hi;
    asm volatile("mfence; rdtsc; lfence" : "=a"(lo), "=d"(hi));
    return ((u64)hi << 32) | lo;
}

#endif /* __KERNEL__ */

#endif /* _AACO_DRIVER_H */
