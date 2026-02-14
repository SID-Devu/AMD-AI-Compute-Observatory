/*
 * AACO Driver User-space Header
 * Shared definitions between kernel module and user-space
 *
 * Author: Sudheer Devu
 * License: GPL v2 / MIT (for user-space usage)
 */

#ifndef _AACO_DRIVER_H
#define _AACO_DRIVER_H

#include <linux/ioctl.h>
#include <linux/types.h>

/* ============================================================================
 * Version and Magic Numbers
 * ============================================================================ */

#define AACO_VERSION_MAJOR  1
#define AACO_VERSION_MINOR  0
#define AACO_VERSION_PATCH  0

#define AACO_MAGIC  'A'

/* ============================================================================
 * Event Types
 * ============================================================================ */

enum aaco_event_type {
    /* Session lifecycle */
    AACO_EVENT_SESSION_START    = 0x0001,
    AACO_EVENT_SESSION_STOP     = 0x0002,
    
    /* Context switches */
    AACO_EVENT_CTX_SWITCH_VOL   = 0x0010,   /* Voluntary */
    AACO_EVENT_CTX_SWITCH_INVOL = 0x0011,   /* Involuntary */
    
    /* Page faults */
    AACO_EVENT_PAGE_FAULT_MAJOR = 0x0020,
    AACO_EVENT_PAGE_FAULT_MINOR = 0x0021,
    
    /* CPU time */
    AACO_EVENT_CPU_TIME         = 0x0030,   /* value1=utime_delta, value2=stime_delta */
    
    /* Memory */
    AACO_EVENT_RSS_SAMPLE       = 0x0040,   /* value1=bytes, value2=MB */
    
    /* Run queue / scheduling latency */
    AACO_EVENT_RUNQ_SAMPLE      = 0x0050,
    AACO_EVENT_SCHED_LATENCY    = 0x0051,
    
    /* User-space markers */
    AACO_EVENT_USER_MARKER      = 0x0100,
    
    /* Phase markers (from user space) */
    AACO_EVENT_PHASE_WARMUP     = 0x0110,
    AACO_EVENT_PHASE_MEASURE    = 0x0111,
    AACO_EVENT_PHASE_PREFILL    = 0x0112,
    AACO_EVENT_PHASE_DECODE     = 0x0113,
    
    /* Inference iteration markers */
    AACO_EVENT_ITER_START       = 0x0120,
    AACO_EVENT_ITER_END         = 0x0121,
    
    /* GPU markers (from user space, for correlation) */
    AACO_EVENT_GPU_KERNEL_START = 0x0200,
    AACO_EVENT_GPU_KERNEL_END   = 0x0201,
};

/* ============================================================================
 * IOCTL Structures
 * ============================================================================ */

/* Session start/stop command */
struct aaco_session_cmd {
    __u32 session_id;           /* Session identifier */
    __u32 pid;                  /* Target PID (0 = current process) */
    __u64 flags;                /* Reserved flags */
    union {
        void *data;             /* For GET_STATS: pointer to aaco_stats */
        struct {
            __u32 marker_id;    /* For EMIT_MARKER */
            __u32 marker_value;
        };
    };
};

/* Session statistics (returned by GET_STATS) */
struct aaco_stats {
    __u32 session_id;
    __u32 pid;
    __u64 duration_ns;
    __u64 samples_collected;
    __u64 events_generated;
    
    /* Aggregated counters */
    __u64 total_nvcsw;          /* Total voluntary context switches */
    __u64 total_nivcsw;         /* Total involuntary context switches */
    __u64 total_maj_flt;        /* Total major page faults */
    __u64 total_min_flt;        /* Total minor page faults */
    __u64 total_utime_ns;       /* Total user CPU time */
    __u64 total_stime_ns;       /* Total system CPU time */
    __u64 rss_peak_bytes;       /* Peak RSS */
    
    /* Reserved for future use */
    __u64 reserved[8];
};

/* Event record (read from device) */
struct aaco_event_record {
    __u64 timestamp_ns;         /* Monotonic timestamp */
    __u32 session_id;           /* Session identifier */
    __u32 pid;                  /* Process ID */
    __u16 event_type;           /* Event type enum */
    __u16 cpu;                  /* CPU where event occurred */
    __u64 value1;               /* Primary value */
    __u64 value2;               /* Secondary value */
    char comm[16];              /* Process name */
} __attribute__((packed));

/* ============================================================================
 * IOCTL Commands
 * ============================================================================ */

/* Start a new tracking session */
#define AACO_IOC_SESSION_START  _IOW(AACO_MAGIC, 0x01, struct aaco_session_cmd)

/* Stop a tracking session */
#define AACO_IOC_SESSION_STOP   _IOW(AACO_MAGIC, 0x02, struct aaco_session_cmd)

/* Get session statistics */
#define AACO_IOC_GET_STATS      _IOWR(AACO_MAGIC, 0x03, struct aaco_session_cmd)

/* Set sampling interval (arg = milliseconds) */
#define AACO_IOC_SET_SAMPLE_MS  _IOW(AACO_MAGIC, 0x04, unsigned long)

/* Emit a user-space marker event */
#define AACO_IOC_EMIT_MARKER    _IOW(AACO_MAGIC, 0x05, struct aaco_session_cmd)

/* Get driver version */
#define AACO_IOC_GET_VERSION    _IOR(AACO_MAGIC, 0x10, unsigned long)

/* ============================================================================
 * Helper Macros
 * ============================================================================ */

#define AACO_DEVICE_PATH    "/dev/aaco"
#define AACO_DEBUGFS_PATH   "/sys/kernel/debug/aaco"

/* Event type category helpers */
#define AACO_EVENT_IS_SESSION(t)    (((t) & 0xFF00) == 0x0000)
#define AACO_EVENT_IS_SCHED(t)      (((t) & 0xFF00) == 0x0010)
#define AACO_EVENT_IS_MEMORY(t)     (((t) & 0xFF00) == 0x0020 || ((t) & 0xFF00) == 0x0040)
#define AACO_EVENT_IS_CPU(t)        (((t) & 0xFF00) == 0x0030)
#define AACO_EVENT_IS_MARKER(t)     (((t) & 0xFF00) == 0x0100)
#define AACO_EVENT_IS_GPU(t)        (((t) & 0xFF00) == 0x0200)

#endif /* _AACO_DRIVER_H */
