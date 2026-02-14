/*
 * AACO Linux Kernel Driver
 * AMD AI Compute Observatory - Kernel-level Observability Module
 *
 * Provides:
 * - /dev/aaco character device for session control
 * - Per-process telemetry collection (ctx switches, faults, CPU time)
 * - High-throughput ring buffer for event streaming
 * - Integration with userspace analytics pipeline
 *
 * Author: Sudheer Devu
 * License: GPL v2
 */

#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/fs.h>
#include <linux/cdev.h>
#include <linux/device.h>
#include <linux/uaccess.h>
#include <linux/slab.h>
#include <linux/spinlock.h>
#include <linux/timer.h>
#include <linux/hrtimer.h>
#include <linux/ktime.h>
#include <linux/sched.h>
#include <linux/sched/signal.h>
#include <linux/mm.h>
#include <linux/debugfs.h>
#include <linux/circ_buf.h>
#include <linux/poll.h>
#include <linux/wait.h>

#include "aaco_driver.h"

#define DRIVER_NAME "aaco"
#define DRIVER_CLASS "aaco_class"
#define DEVICE_NAME "aaco"

/* Ring buffer size (must be power of 2) */
#define AACO_RING_SIZE (1 << 16)  /* 64K events */
#define AACO_RING_MASK (AACO_RING_SIZE - 1)

/* Maximum concurrent sessions */
#define AACO_MAX_SESSIONS 64

/* Sampling timer default interval (ms) */
#define AACO_DEFAULT_SAMPLE_MS 10

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Sudheer Devu");
MODULE_DESCRIPTION("AMD AI Compute Observatory - Kernel Observability Driver");
MODULE_VERSION("1.0");

/* ============================================================================
 * Data Structures
 * ============================================================================ */

/* Event record structure (packed for efficient transfer) */
struct aaco_event {
    u64 timestamp_ns;           /* Monotonic timestamp */
    u32 session_id;             /* Session identifier */
    u32 pid;                    /* Process ID */
    u16 event_type;             /* Event type enum */
    u16 cpu;                    /* CPU where event occurred */
    u64 value1;                 /* Primary value */
    u64 value2;                 /* Secondary value */
    char comm[16];              /* Process name */
} __attribute__((packed));

/* Per-session tracking context */
struct aaco_session {
    u32 session_id;
    pid_t pid;
    pid_t tgid;
    
    /* Baseline counters at session start */
    u64 start_time_ns;
    u64 start_nvcsw;            /* Voluntary context switches */
    u64 start_nivcsw;           /* Involuntary context switches */
    u64 start_maj_flt;          /* Major page faults */
    u64 start_min_flt;          /* Minor page faults */
    u64 start_utime;            /* User CPU time */
    u64 start_stime;            /* System CPU time */
    
    /* Current sampled values */
    u64 last_nvcsw;
    u64 last_nivcsw;
    u64 last_maj_flt;
    u64 last_min_flt;
    u64 last_utime;
    u64 last_stime;
    u64 last_rss;
    
    /* Sampling configuration */
    u32 sample_interval_ms;
    bool active;
    
    /* High-resolution timer for sampling */
    struct hrtimer sample_timer;
    
    /* Statistics */
    u64 events_generated;
    u64 samples_taken;
};

/* Ring buffer for events */
struct aaco_ring_buffer {
    struct aaco_event *events;
    unsigned int head;          /* Write position */
    unsigned int tail;          /* Read position */
    spinlock_t lock;
    wait_queue_head_t wait_queue;
    bool overflow;
    u64 overflow_count;
};

/* Driver global state */
struct aaco_driver_state {
    dev_t dev_num;
    struct cdev cdev;
    struct class *dev_class;
    struct device *device;
    
    /* Sessions */
    struct aaco_session *sessions[AACO_MAX_SESSIONS];
    int session_count;
    struct mutex sessions_lock;
    
    /* Ring buffer */
    struct aaco_ring_buffer ring;
    
    /* Debugfs */
    struct dentry *debugfs_dir;
    
    /* Statistics */
    u64 total_events;
    u64 total_sessions;
    u64 total_reads;
};

static struct aaco_driver_state *aaco_state;

/* ============================================================================
 * Ring Buffer Operations
 * ============================================================================ */

static int aaco_ring_init(struct aaco_ring_buffer *ring)
{
    ring->events = vzalloc(AACO_RING_SIZE * sizeof(struct aaco_event));
    if (!ring->events)
        return -ENOMEM;
    
    ring->head = 0;
    ring->tail = 0;
    ring->overflow = false;
    ring->overflow_count = 0;
    spin_lock_init(&ring->lock);
    init_waitqueue_head(&ring->wait_queue);
    
    return 0;
}

static void aaco_ring_destroy(struct aaco_ring_buffer *ring)
{
    if (ring->events) {
        vfree(ring->events);
        ring->events = NULL;
    }
}

static bool aaco_ring_is_empty(struct aaco_ring_buffer *ring)
{
    return ring->head == ring->tail;
}

static bool aaco_ring_is_full(struct aaco_ring_buffer *ring)
{
    return ((ring->head + 1) & AACO_RING_MASK) == ring->tail;
}

static int aaco_ring_write(struct aaco_ring_buffer *ring, 
                           const struct aaco_event *event)
{
    unsigned long flags;
    
    spin_lock_irqsave(&ring->lock, flags);
    
    if (aaco_ring_is_full(ring)) {
        ring->overflow = true;
        ring->overflow_count++;
        spin_unlock_irqrestore(&ring->lock, flags);
        return -ENOSPC;
    }
    
    memcpy(&ring->events[ring->head], event, sizeof(*event));
    ring->head = (ring->head + 1) & AACO_RING_MASK;
    
    spin_unlock_irqrestore(&ring->lock, flags);
    
    wake_up_interruptible(&ring->wait_queue);
    
    return 0;
}

static int aaco_ring_read(struct aaco_ring_buffer *ring,
                          struct aaco_event *event)
{
    unsigned long flags;
    
    spin_lock_irqsave(&ring->lock, flags);
    
    if (aaco_ring_is_empty(ring)) {
        spin_unlock_irqrestore(&ring->lock, flags);
        return -EAGAIN;
    }
    
    memcpy(event, &ring->events[ring->tail], sizeof(*event));
    ring->tail = (ring->tail + 1) & AACO_RING_MASK;
    
    spin_unlock_irqrestore(&ring->lock, flags);
    
    return 0;
}

static size_t aaco_ring_count(struct aaco_ring_buffer *ring)
{
    return (ring->head - ring->tail) & AACO_RING_MASK;
}

/* ============================================================================
 * Event Generation
 * ============================================================================ */

static void aaco_emit_event(struct aaco_session *session, u16 event_type,
                            u64 value1, u64 value2, const char *comm)
{
    struct aaco_event event;
    
    event.timestamp_ns = ktime_get_ns();
    event.session_id = session->session_id;
    event.pid = session->pid;
    event.event_type = event_type;
    event.cpu = smp_processor_id();
    event.value1 = value1;
    event.value2 = value2;
    
    if (comm) {
        strncpy(event.comm, comm, sizeof(event.comm) - 1);
        event.comm[sizeof(event.comm) - 1] = '\0';
    } else {
        event.comm[0] = '\0';
    }
    
    if (aaco_ring_write(&aaco_state->ring, &event) == 0) {
        session->events_generated++;
        aaco_state->total_events++;
    }
}

/* ============================================================================
 * Process Sampling
 * ============================================================================ */

static struct task_struct *find_task_by_pid(pid_t pid)
{
    struct task_struct *task;
    
    rcu_read_lock();
    task = pid_task(find_vpid(pid), PIDTYPE_PID);
    if (task)
        get_task_struct(task);
    rcu_read_unlock();
    
    return task;
}

static void aaco_sample_process(struct aaco_session *session)
{
    struct task_struct *task;
    u64 nvcsw, nivcsw, maj_flt, min_flt, utime, stime, rss;
    u64 d_nvcsw, d_nivcsw, d_maj_flt, d_min_flt;
    struct mm_struct *mm;
    
    task = find_task_by_pid(session->pid);
    if (!task)
        return;
    
    /* Read current counters */
    nvcsw = task->nvcsw;
    nivcsw = task->nivcsw;
    maj_flt = task->maj_flt;
    min_flt = task->min_flt;
    utime = task->utime;
    stime = task->stime;
    
    /* Get RSS */
    mm = get_task_mm(task);
    if (mm) {
        rss = get_mm_rss(mm) << PAGE_SHIFT;
        mmput(mm);
    } else {
        rss = 0;
    }
    
    /* Calculate deltas since last sample */
    d_nvcsw = nvcsw - session->last_nvcsw;
    d_nivcsw = nivcsw - session->last_nivcsw;
    d_maj_flt = maj_flt - session->last_maj_flt;
    d_min_flt = min_flt - session->last_min_flt;
    
    /* Emit events for significant changes */
    if (d_nvcsw > 0) {
        aaco_emit_event(session, AACO_EVENT_CTX_SWITCH_VOL, 
                        d_nvcsw, nvcsw, task->comm);
    }
    
    if (d_nivcsw > 0) {
        aaco_emit_event(session, AACO_EVENT_CTX_SWITCH_INVOL,
                        d_nivcsw, nivcsw, task->comm);
    }
    
    if (d_maj_flt > 0) {
        aaco_emit_event(session, AACO_EVENT_PAGE_FAULT_MAJOR,
                        d_maj_flt, maj_flt, task->comm);
    }
    
    if (d_min_flt > 0) {
        aaco_emit_event(session, AACO_EVENT_PAGE_FAULT_MINOR,
                        d_min_flt, min_flt, task->comm);
    }
    
    /* Emit CPU time sample */
    aaco_emit_event(session, AACO_EVENT_CPU_TIME,
                    utime - session->last_utime,
                    stime - session->last_stime, task->comm);
    
    /* Emit RSS sample */
    aaco_emit_event(session, AACO_EVENT_RSS_SAMPLE,
                    rss, rss >> 20, task->comm);  /* value2 = MB */
    
    /* Update last values */
    session->last_nvcsw = nvcsw;
    session->last_nivcsw = nivcsw;
    session->last_maj_flt = maj_flt;
    session->last_min_flt = min_flt;
    session->last_utime = utime;
    session->last_stime = stime;
    session->last_rss = rss;
    session->samples_taken++;
    
    put_task_struct(task);
}

/* ============================================================================
 * High-Resolution Timer Callback
 * ============================================================================ */

static enum hrtimer_restart aaco_timer_callback(struct hrtimer *timer)
{
    struct aaco_session *session = container_of(timer, struct aaco_session, 
                                                sample_timer);
    ktime_t interval;
    
    if (!session->active)
        return HRTIMER_NORESTART;
    
    /* Sample the process */
    aaco_sample_process(session);
    
    /* Reschedule timer */
    interval = ms_to_ktime(session->sample_interval_ms);
    hrtimer_forward_now(timer, interval);
    
    return HRTIMER_RESTART;
}

/* ============================================================================
 * Session Management
 * ============================================================================ */

static struct aaco_session *aaco_session_create(u32 session_id, pid_t pid)
{
    struct aaco_session *session;
    struct task_struct *task;
    
    session = kzalloc(sizeof(*session), GFP_KERNEL);
    if (!session)
        return ERR_PTR(-ENOMEM);
    
    session->session_id = session_id;
    session->pid = pid;
    session->sample_interval_ms = AACO_DEFAULT_SAMPLE_MS;
    session->active = false;
    
    /* Get initial counters */
    task = find_task_by_pid(pid);
    if (!task) {
        kfree(session);
        return ERR_PTR(-ESRCH);
    }
    
    session->tgid = task->tgid;
    session->start_nvcsw = task->nvcsw;
    session->start_nivcsw = task->nivcsw;
    session->start_maj_flt = task->maj_flt;
    session->start_min_flt = task->min_flt;
    session->start_utime = task->utime;
    session->start_stime = task->stime;
    
    session->last_nvcsw = session->start_nvcsw;
    session->last_nivcsw = session->start_nivcsw;
    session->last_maj_flt = session->start_maj_flt;
    session->last_min_flt = session->start_min_flt;
    session->last_utime = session->start_utime;
    session->last_stime = session->start_stime;
    
    put_task_struct(task);
    
    /* Initialize timer */
    hrtimer_init(&session->sample_timer, CLOCK_MONOTONIC, HRTIMER_MODE_REL);
    session->sample_timer.function = aaco_timer_callback;
    
    return session;
}

static void aaco_session_destroy(struct aaco_session *session)
{
    if (!session)
        return;
    
    if (session->active) {
        session->active = false;
        hrtimer_cancel(&session->sample_timer);
    }
    
    kfree(session);
}

static int aaco_session_start(struct aaco_session *session)
{
    ktime_t interval;
    
    if (session->active)
        return -EALREADY;
    
    session->start_time_ns = ktime_get_ns();
    session->active = true;
    
    /* Emit session start event */
    aaco_emit_event(session, AACO_EVENT_SESSION_START,
                    session->session_id, session->pid, current->comm);
    
    /* Start sampling timer */
    interval = ms_to_ktime(session->sample_interval_ms);
    hrtimer_start(&session->sample_timer, interval, HRTIMER_MODE_REL);
    
    pr_info("aaco: session %u started for pid %d\n",
            session->session_id, session->pid);
    
    return 0;
}

static int aaco_session_stop(struct aaco_session *session)
{
    u64 duration_ns;
    
    if (!session->active)
        return -EINVAL;
    
    session->active = false;
    hrtimer_cancel(&session->sample_timer);
    
    duration_ns = ktime_get_ns() - session->start_time_ns;
    
    /* Emit session stop event */
    aaco_emit_event(session, AACO_EVENT_SESSION_STOP,
                    duration_ns, session->samples_taken, current->comm);
    
    pr_info("aaco: session %u stopped (duration: %llu ns, samples: %llu)\n",
            session->session_id, duration_ns, session->samples_taken);
    
    return 0;
}

static int aaco_session_get_stats(struct aaco_session *session,
                                  struct aaco_stats __user *user_stats)
{
    struct aaco_stats stats;
    struct task_struct *task;
    
    memset(&stats, 0, sizeof(stats));
    
    stats.session_id = session->session_id;
    stats.pid = session->pid;
    stats.duration_ns = session->active ? 
                        (ktime_get_ns() - session->start_time_ns) : 0;
    stats.samples_collected = session->samples_taken;
    stats.events_generated = session->events_generated;
    
    task = find_task_by_pid(session->pid);
    if (task) {
        stats.total_nvcsw = task->nvcsw - session->start_nvcsw;
        stats.total_nivcsw = task->nivcsw - session->start_nivcsw;
        stats.total_maj_flt = task->maj_flt - session->start_maj_flt;
        stats.total_min_flt = task->min_flt - session->start_min_flt;
        stats.total_utime_ns = task->utime - session->start_utime;
        stats.total_stime_ns = task->stime - session->start_stime;
        stats.rss_peak_bytes = session->last_rss;  /* Simplified */
        put_task_struct(task);
    }
    
    if (copy_to_user(user_stats, &stats, sizeof(stats)))
        return -EFAULT;
    
    return 0;
}

/* ============================================================================
 * File Operations
 * ============================================================================ */

static int aaco_open(struct inode *inode, struct file *file)
{
    /* Per-file state could be added here */
    file->private_data = NULL;
    return 0;
}

static int aaco_release(struct inode *inode, struct file *file)
{
    return 0;
}

static ssize_t aaco_read(struct file *file, char __user *buf,
                         size_t count, loff_t *ppos)
{
    struct aaco_event event;
    size_t events_read = 0;
    size_t events_to_read = count / sizeof(struct aaco_event);
    int ret;
    
    if (events_to_read == 0)
        return -EINVAL;
    
    /* Read available events */
    while (events_read < events_to_read) {
        ret = aaco_ring_read(&aaco_state->ring, &event);
        if (ret == -EAGAIN) {
            if (events_read > 0)
                break;
            
            /* No events available */
            if (file->f_flags & O_NONBLOCK)
                return -EAGAIN;
            
            /* Wait for events */
            ret = wait_event_interruptible(aaco_state->ring.wait_queue,
                                           !aaco_ring_is_empty(&aaco_state->ring));
            if (ret)
                return ret;
            
            continue;
        }
        
        if (copy_to_user(buf + events_read * sizeof(event),
                         &event, sizeof(event)))
            return -EFAULT;
        
        events_read++;
    }
    
    aaco_state->total_reads++;
    
    return events_read * sizeof(struct aaco_event);
}

static unsigned int aaco_poll(struct file *file, poll_table *wait)
{
    unsigned int mask = 0;
    
    poll_wait(file, &aaco_state->ring.wait_queue, wait);
    
    if (!aaco_ring_is_empty(&aaco_state->ring))
        mask |= POLLIN | POLLRDNORM;
    
    return mask;
}

static long aaco_ioctl(struct file *file, unsigned int cmd, unsigned long arg)
{
    struct aaco_session_cmd session_cmd;
    struct aaco_session *session;
    int ret = 0;
    int i;
    
    switch (cmd) {
    case AACO_IOC_SESSION_START:
        if (copy_from_user(&session_cmd, (void __user *)arg, sizeof(session_cmd)))
            return -EFAULT;
        
        mutex_lock(&aaco_state->sessions_lock);
        
        /* Check if session already exists */
        for (i = 0; i < AACO_MAX_SESSIONS; i++) {
            if (aaco_state->sessions[i] &&
                aaco_state->sessions[i]->session_id == session_cmd.session_id) {
                mutex_unlock(&aaco_state->sessions_lock);
                return -EEXIST;
            }
        }
        
        /* Find empty slot */
        for (i = 0; i < AACO_MAX_SESSIONS; i++) {
            if (!aaco_state->sessions[i])
                break;
        }
        
        if (i >= AACO_MAX_SESSIONS) {
            mutex_unlock(&aaco_state->sessions_lock);
            return -ENOSPC;
        }
        
        session = aaco_session_create(session_cmd.session_id, 
                                      session_cmd.pid ? session_cmd.pid : current->pid);
        if (IS_ERR(session)) {
            mutex_unlock(&aaco_state->sessions_lock);
            return PTR_ERR(session);
        }
        
        aaco_state->sessions[i] = session;
        aaco_state->session_count++;
        aaco_state->total_sessions++;
        
        ret = aaco_session_start(session);
        
        mutex_unlock(&aaco_state->sessions_lock);
        break;
        
    case AACO_IOC_SESSION_STOP:
        if (copy_from_user(&session_cmd, (void __user *)arg, sizeof(session_cmd)))
            return -EFAULT;
        
        mutex_lock(&aaco_state->sessions_lock);
        
        session = NULL;
        for (i = 0; i < AACO_MAX_SESSIONS; i++) {
            if (aaco_state->sessions[i] &&
                aaco_state->sessions[i]->session_id == session_cmd.session_id) {
                session = aaco_state->sessions[i];
                break;
            }
        }
        
        if (!session) {
            mutex_unlock(&aaco_state->sessions_lock);
            return -ENOENT;
        }
        
        ret = aaco_session_stop(session);
        aaco_session_destroy(session);
        aaco_state->sessions[i] = NULL;
        aaco_state->session_count--;
        
        mutex_unlock(&aaco_state->sessions_lock);
        break;
        
    case AACO_IOC_GET_STATS:
        if (copy_from_user(&session_cmd, (void __user *)arg, sizeof(session_cmd)))
            return -EFAULT;
        
        mutex_lock(&aaco_state->sessions_lock);
        
        session = NULL;
        for (i = 0; i < AACO_MAX_SESSIONS; i++) {
            if (aaco_state->sessions[i] &&
                aaco_state->sessions[i]->session_id == session_cmd.session_id) {
                session = aaco_state->sessions[i];
                break;
            }
        }
        
        if (!session) {
            mutex_unlock(&aaco_state->sessions_lock);
            return -ENOENT;
        }
        
        ret = aaco_session_get_stats(session, 
                                     (struct aaco_stats __user *)session_cmd.data);
        
        mutex_unlock(&aaco_state->sessions_lock);
        break;
        
    case AACO_IOC_SET_SAMPLE_MS:
        if (arg < 1 || arg > 1000)
            return -EINVAL;
        
        mutex_lock(&aaco_state->sessions_lock);
        
        /* Set for all active sessions */
        for (i = 0; i < AACO_MAX_SESSIONS; i++) {
            if (aaco_state->sessions[i])
                aaco_state->sessions[i]->sample_interval_ms = arg;
        }
        
        mutex_unlock(&aaco_state->sessions_lock);
        break;
        
    case AACO_IOC_EMIT_MARKER:
        /* User-space marker emission */
        if (copy_from_user(&session_cmd, (void __user *)arg, sizeof(session_cmd)))
            return -EFAULT;
        
        mutex_lock(&aaco_state->sessions_lock);
        
        session = NULL;
        for (i = 0; i < AACO_MAX_SESSIONS; i++) {
            if (aaco_state->sessions[i] &&
                aaco_state->sessions[i]->session_id == session_cmd.session_id) {
                session = aaco_state->sessions[i];
                break;
            }
        }
        
        if (session) {
            aaco_emit_event(session, AACO_EVENT_USER_MARKER,
                           session_cmd.marker_id, session_cmd.marker_value, NULL);
        }
        
        mutex_unlock(&aaco_state->sessions_lock);
        break;
        
    default:
        return -EINVAL;
    }
    
    return ret;
}

static const struct file_operations aaco_fops = {
    .owner = THIS_MODULE,
    .open = aaco_open,
    .release = aaco_release,
    .read = aaco_read,
    .poll = aaco_poll,
    .unlocked_ioctl = aaco_ioctl,
    .compat_ioctl = aaco_ioctl,
};

/* ============================================================================
 * Debugfs Interface
 * ============================================================================ */

static int aaco_debugfs_stats_show(struct seq_file *m, void *v)
{
    seq_printf(m, "AACO Driver Statistics\n");
    seq_printf(m, "=======================\n");
    seq_printf(m, "Total sessions created: %llu\n", aaco_state->total_sessions);
    seq_printf(m, "Active sessions: %d\n", aaco_state->session_count);
    seq_printf(m, "Total events generated: %llu\n", aaco_state->total_events);
    seq_printf(m, "Total reads: %llu\n", aaco_state->total_reads);
    seq_printf(m, "Ring buffer size: %d\n", AACO_RING_SIZE);
    seq_printf(m, "Ring buffer usage: %zu\n", aaco_ring_count(&aaco_state->ring));
    seq_printf(m, "Ring buffer overflows: %llu\n", aaco_state->ring.overflow_count);
    
    return 0;
}

static int aaco_debugfs_stats_open(struct inode *inode, struct file *file)
{
    return single_open(file, aaco_debugfs_stats_show, NULL);
}

static const struct file_operations aaco_debugfs_stats_fops = {
    .owner = THIS_MODULE,
    .open = aaco_debugfs_stats_open,
    .read = seq_read,
    .llseek = seq_lseek,
    .release = single_release,
};

static int aaco_debugfs_sessions_show(struct seq_file *m, void *v)
{
    int i;
    struct aaco_session *s;
    
    seq_printf(m, "Active Sessions\n");
    seq_printf(m, "===============\n");
    
    mutex_lock(&aaco_state->sessions_lock);
    
    for (i = 0; i < AACO_MAX_SESSIONS; i++) {
        s = aaco_state->sessions[i];
        if (s) {
            seq_printf(m, "Session %u: pid=%d, samples=%llu, events=%llu, interval=%ums\n",
                      s->session_id, s->pid, s->samples_taken,
                      s->events_generated, s->sample_interval_ms);
        }
    }
    
    mutex_unlock(&aaco_state->sessions_lock);
    
    return 0;
}

static int aaco_debugfs_sessions_open(struct inode *inode, struct file *file)
{
    return single_open(file, aaco_debugfs_sessions_show, NULL);
}

static const struct file_operations aaco_debugfs_sessions_fops = {
    .owner = THIS_MODULE,
    .open = aaco_debugfs_sessions_open,
    .read = seq_read,
    .llseek = seq_lseek,
    .release = single_release,
};

static void aaco_debugfs_init(void)
{
    aaco_state->debugfs_dir = debugfs_create_dir("aaco", NULL);
    if (!aaco_state->debugfs_dir) {
        pr_warn("aaco: failed to create debugfs directory\n");
        return;
    }
    
    debugfs_create_file("stats", 0444, aaco_state->debugfs_dir,
                        NULL, &aaco_debugfs_stats_fops);
    debugfs_create_file("sessions", 0444, aaco_state->debugfs_dir,
                        NULL, &aaco_debugfs_sessions_fops);
}

static void aaco_debugfs_cleanup(void)
{
    if (aaco_state->debugfs_dir)
        debugfs_remove_recursive(aaco_state->debugfs_dir);
}

/* ============================================================================
 * Module Init/Exit
 * ============================================================================ */

static int __init aaco_init(void)
{
    int ret;
    
    pr_info("aaco: initializing AMD AI Compute Observatory driver\n");
    
    /* Allocate driver state */
    aaco_state = kzalloc(sizeof(*aaco_state), GFP_KERNEL);
    if (!aaco_state)
        return -ENOMEM;
    
    mutex_init(&aaco_state->sessions_lock);
    
    /* Initialize ring buffer */
    ret = aaco_ring_init(&aaco_state->ring);
    if (ret) {
        pr_err("aaco: failed to initialize ring buffer\n");
        goto err_free_state;
    }
    
    /* Allocate device number */
    ret = alloc_chrdev_region(&aaco_state->dev_num, 0, 1, DEVICE_NAME);
    if (ret < 0) {
        pr_err("aaco: failed to allocate device number\n");
        goto err_ring_destroy;
    }
    
    /* Initialize character device */
    cdev_init(&aaco_state->cdev, &aaco_fops);
    aaco_state->cdev.owner = THIS_MODULE;
    
    ret = cdev_add(&aaco_state->cdev, aaco_state->dev_num, 1);
    if (ret < 0) {
        pr_err("aaco: failed to add character device\n");
        goto err_unregister;
    }
    
    /* Create device class */
    aaco_state->dev_class = class_create(THIS_MODULE, DRIVER_CLASS);
    if (IS_ERR(aaco_state->dev_class)) {
        pr_err("aaco: failed to create device class\n");
        ret = PTR_ERR(aaco_state->dev_class);
        goto err_cdev_del;
    }
    
    /* Create device */
    aaco_state->device = device_create(aaco_state->dev_class, NULL,
                                       aaco_state->dev_num, NULL, DEVICE_NAME);
    if (IS_ERR(aaco_state->device)) {
        pr_err("aaco: failed to create device\n");
        ret = PTR_ERR(aaco_state->device);
        goto err_class_destroy;
    }
    
    /* Initialize debugfs */
    aaco_debugfs_init();
    
    pr_info("aaco: driver initialized successfully (major=%d)\n",
            MAJOR(aaco_state->dev_num));
    
    return 0;

err_class_destroy:
    class_destroy(aaco_state->dev_class);
err_cdev_del:
    cdev_del(&aaco_state->cdev);
err_unregister:
    unregister_chrdev_region(aaco_state->dev_num, 1);
err_ring_destroy:
    aaco_ring_destroy(&aaco_state->ring);
err_free_state:
    kfree(aaco_state);
    return ret;
}

static void __exit aaco_exit(void)
{
    int i;
    
    pr_info("aaco: unloading driver\n");
    
    /* Stop and destroy all sessions */
    mutex_lock(&aaco_state->sessions_lock);
    for (i = 0; i < AACO_MAX_SESSIONS; i++) {
        if (aaco_state->sessions[i]) {
            aaco_session_stop(aaco_state->sessions[i]);
            aaco_session_destroy(aaco_state->sessions[i]);
            aaco_state->sessions[i] = NULL;
        }
    }
    mutex_unlock(&aaco_state->sessions_lock);
    
    /* Cleanup debugfs */
    aaco_debugfs_cleanup();
    
    /* Cleanup device */
    device_destroy(aaco_state->dev_class, aaco_state->dev_num);
    class_destroy(aaco_state->dev_class);
    cdev_del(&aaco_state->cdev);
    unregister_chrdev_region(aaco_state->dev_num, 1);
    
    /* Cleanup ring buffer */
    aaco_ring_destroy(&aaco_state->ring);
    
    /* Free state */
    kfree(aaco_state);
    
    pr_info("aaco: driver unloaded\n");
}

module_init(aaco_init);
module_exit(aaco_exit);
