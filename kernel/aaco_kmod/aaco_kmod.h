/* SPDX-License-Identifier: GPL-2.0 */
/*
 * AACO Kernel Module Header
 * 
 * Public interface for the AACO profiling kernel module.
 */

#ifndef _AACO_KMOD_H
#define _AACO_KMOD_H

#include <linux/types.h>
#include <linux/ioctl.h>

#define AACO_KMOD_VERSION "1.0.0"

/* IOCTL definitions */
#define AACO_IOC_MAGIC    'A'
#define AACO_IOC_START    _IO(AACO_IOC_MAGIC, 0)
#define AACO_IOC_STOP     _IO(AACO_IOC_MAGIC, 1)
#define AACO_IOC_READ     _IOR(AACO_IOC_MAGIC, 2, struct aaco_sample)
#define AACO_IOC_CONFIG   _IOW(AACO_IOC_MAGIC, 3, struct aaco_config)
#define AACO_IOC_STATUS   _IOR(AACO_IOC_MAGIC, 4, struct aaco_status)

/* Sample types */
#define AACO_SAMPLE_COUNTER    0x01
#define AACO_SAMPLE_TIMESTAMP  0x02
#define AACO_SAMPLE_MEMORY     0x03
#define AACO_SAMPLE_POWER      0x04
#define AACO_SAMPLE_THERMAL    0x05

/* Configuration flags */
#define AACO_FLAG_COUNTERS     0x01
#define AACO_FLAG_TIMESTAMPS   0x02
#define AACO_FLAG_MEMORY       0x04
#define AACO_FLAG_POWER        0x08
#define AACO_FLAG_THERMAL      0x10
#define AACO_FLAG_ALL          0xFF

/**
 * struct aaco_sample - Single profiling sample
 * @timestamp: Nanosecond timestamp (CLOCK_MONOTONIC)
 * @gpu_id: GPU device ID
 * @type: Sample type (AACO_SAMPLE_*)
 * @value: Sample value
 * @metadata: Additional metadata (type-dependent)
 */
struct aaco_sample {
    __u64 timestamp;
    __u32 gpu_id;
    __u32 type;
    __u64 value;
    __u64 metadata;
};

/**
 * struct aaco_config - Profiling configuration
 * @sample_interval_us: Sampling interval in microseconds
 * @gpu_mask: Bitmask of GPUs to profile
 * @counter_mask: Bitmask of counters to collect
 * @flags: Configuration flags (AACO_FLAG_*)
 */
struct aaco_config {
    __u32 sample_interval_us;
    __u32 gpu_mask;
    __u32 counter_mask;
    __u32 flags;
};

/**
 * struct aaco_status - Module status
 * @active: Whether profiling is active
 * @samples_collected: Total samples collected
 * @samples_dropped: Samples dropped due to buffer overflow
 * @gpu_count: Number of GPUs detected
 */
struct aaco_status {
    __u32 active;
    __u64 samples_collected;
    __u64 samples_dropped;
    __u32 gpu_count;
};

/* Kernel-space API (for other modules) */
#ifdef __KERNEL__

/**
 * aaco_add_sample - Add a sample to the buffer
 * @sample: Sample to add
 *
 * Called by other kernel components to submit profiling samples.
 * Safe to call from interrupt context.
 */
void aaco_add_sample(struct aaco_sample *sample);

#endif /* __KERNEL__ */

#endif /* _AACO_KMOD_H */
