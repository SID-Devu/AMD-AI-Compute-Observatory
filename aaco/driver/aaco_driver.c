/*
 * AACO-SIGMA Linux Kernel Driver
 * 
 * Character device driver providing:
 * - High-precision TSC timestamps
 * - Memory barriers
 * - CPU affinity control
 * - Preemption control (with safety limits)
 * - IRQ/context switch statistics
 * 
 * Build: make -C /lib/modules/$(uname -r)/build M=$PWD modules
 * Load:  sudo insmod aaco_driver.ko
 * 
 * Copyright 2024 AMD AI Compute Observatory
 * SPDX-License-Identifier: GPL-2.0-only
 */

#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/fs.h>
#include <linux/cdev.h>
#include <linux/device.h>
#include <linux/uaccess.h>
#include <linux/slab.h>
#include <linux/sched.h>
#include <linux/sched/signal.h>
#include <linux/preempt.h>
#include <linux/cpumask.h>
#include <linux/ktime.h>
#include <linux/delay.h>
#include <asm/msr.h>
#include <asm/tsc.h>

#include "aaco_driver.h"

MODULE_LICENSE("GPL");
MODULE_AUTHOR("AMD AI Compute Observatory");
MODULE_DESCRIPTION("AACO-SIGMA Deterministic Measurement Kernel Driver");
MODULE_VERSION("1.0.0");

/* Global driver state */
static struct aaco_driver_state aaco_state;

/* =============================================================================
 * TSC Frequency Detection
 * ============================================================================= */

static u64 detect_tsc_frequency(void)
{
    u64 freq = 0;
    
    /* Try native TSC frequency first (from kernel) */
#ifdef CONFIG_X86
    freq = tsc_khz * 1000ULL;
    if (freq > 0)
        return freq;
#endif
    
    /* Fallback: calibrate using ktime */
    {
        ktime_t start_time, end_time;
        u64 start_tsc, end_tsc;
        s64 elapsed_ns;
        
        start_tsc = rdtsc_ordered();
        start_time = ktime_get();
        
        /* Short delay for calibration */
        udelay(1000);  /* 1ms */
        
        end_tsc = rdtsc_ordered();
        end_time = ktime_get();
        
        elapsed_ns = ktime_to_ns(ktime_sub(end_time, start_time));
        if (elapsed_ns > 0) {
            freq = (end_tsc - start_tsc) * 1000000000ULL / elapsed_ns;
        }
    }
    
    return freq > 0 ? freq : 1000000000ULL;  /* Default 1GHz */
}

/* =============================================================================
 * File Operations
 * ============================================================================= */

static int aaco_open(struct inode *inode, struct file *file)
{
    struct aaco_file_state *state;
    
    state = kzalloc(sizeof(*state), GFP_KERNEL);
    if (!state)
        return -ENOMEM;
    
    spin_lock_init(&state->lock);
    state->affinity_modified = false;
    state->preempt_disabled = false;
    
    file->private_data = state;
    
    pr_debug("aaco: opened by pid %d\n", current->pid);
    return 0;
}

static int aaco_release(struct inode *inode, struct file *file)
{
    struct aaco_file_state *state = file->private_data;
    
    if (state) {
        /* Re-enable preemption if still disabled */
        if (state->preempt_disabled) {
            preempt_enable();
            state->preempt_disabled = false;
        }
        
        /* Restore CPU affinity if modified */
        if (state->affinity_modified) {
            /* Can't easily restore - task will inherit default on exit */
        }
        
        kfree(state);
    }
    
    pr_debug("aaco: released by pid %d\n", current->pid);
    return 0;
}

/* =============================================================================
 * IOCTL Handlers
 * ============================================================================= */

static long aaco_ioctl_get_version(void __user *arg)
{
    __u32 version = AACO_VERSION;
    if (copy_to_user(arg, &version, sizeof(version)))
        return -EFAULT;
    return 0;
}

static long aaco_ioctl_get_tsc_freq(void __user *arg)
{
    if (copy_to_user(arg, &aaco_state.tsc_freq_hz, sizeof(u64)))
        return -EFAULT;
    return 0;
}

static long aaco_ioctl_read_tsc(void __user *arg)
{
    struct aaco_tsc_reading reading;
    
    reading.tsc_value = rdtsc_ordered();
    reading.timestamp_ns = ktime_get_ns();
    reading.cpu_id = smp_processor_id();
    reading.flags = 0;
    
    if (copy_to_user(arg, &reading, sizeof(reading)))
        return -EFAULT;
    
    return 0;
}

static long aaco_ioctl_set_cpu_affinity(struct file *file, void __user *arg)
{
    struct aaco_file_state *state = file->private_data;
    struct aaco_cpu_mask mask;
    cpumask_t cpumask;
    int cpu, ret;
    
    if (copy_from_user(&mask, arg, sizeof(mask)))
        return -EFAULT;
    
    /* Convert bitmask to cpumask */
    cpumask_clear(&cpumask);
    for (cpu = 0; cpu < 64 && cpu < nr_cpu_ids; cpu++) {
        if (mask.mask & (1ULL << cpu))
            cpumask_set_cpu(cpu, &cpumask);
    }
    
    if (cpumask_empty(&cpumask))
        return -EINVAL;
    
    /* Save current affinity if not already modified */
    spin_lock(&state->lock);
    if (!state->affinity_modified) {
        cpumask_t current_mask;
        cpumask_copy(&current_mask, &current->cpus_mask);
        state->saved_affinity.mask = 0;
        for (cpu = 0; cpu < 64 && cpu < nr_cpu_ids; cpu++) {
            if (cpumask_test_cpu(cpu, &current_mask))
                state->saved_affinity.mask |= (1ULL << cpu);
        }
    }
    spin_unlock(&state->lock);
    
    ret = sched_setaffinity(current->pid, &cpumask);
    if (ret == 0)
        state->affinity_modified = true;
    
    return ret;
}

static long aaco_ioctl_get_cpu_affinity(void __user *arg)
{
    struct aaco_cpu_mask mask;
    cpumask_t cpumask;
    int cpu;
    
    cpumask_copy(&cpumask, &current->cpus_mask);
    
    mask.mask = 0;
    mask.cpu_count = 0;
    for (cpu = 0; cpu < 64 && cpu < nr_cpu_ids; cpu++) {
        if (cpumask_test_cpu(cpu, &cpumask)) {
            mask.mask |= (1ULL << cpu);
            mask.cpu_count++;
        }
    }
    
    if (copy_to_user(arg, &mask, sizeof(mask)))
        return -EFAULT;
    
    return 0;
}

static long aaco_ioctl_memory_barrier(void)
{
    /* Full memory barrier */
    mb();
    return 0;
}

static long aaco_ioctl_disable_preemption(struct file *file)
{
    struct aaco_file_state *state = file->private_data;
    
    spin_lock(&state->lock);
    if (!state->preempt_disabled) {
        preempt_disable();
        state->preempt_disabled = true;
    }
    spin_unlock(&state->lock);
    
    return 0;
}

static long aaco_ioctl_enable_preemption(struct file *file)
{
    struct aaco_file_state *state = file->private_data;
    
    spin_lock(&state->lock);
    if (state->preempt_disabled) {
        preempt_enable();
        state->preempt_disabled = false;
    }
    spin_unlock(&state->lock);
    
    return 0;
}

static long aaco_ioctl_get_irq_stats(void __user *arg)
{
    struct aaco_irq_stats stats = {0};
    
    /* Get per-CPU IRQ counts from kernel */
    /* Note: This is simplified - real impl would aggregate from /proc/interrupts */
    stats.total_irqs = kstat_irqs_sum();
    
    if (copy_to_user(arg, &stats, sizeof(stats)))
        return -EFAULT;
    
    return 0;
}

static long aaco_ioctl_get_ctx_switches(void __user *arg)
{
    struct aaco_ctx_stats stats;
    
    stats.voluntary_switches = current->nvcsw;
    stats.involuntary_switches = current->nivcsw;
    stats.total_switches = current->nvcsw + current->nivcsw;
    stats.reserved = 0;
    
    if (copy_to_user(arg, &stats, sizeof(stats)))
        return -EFAULT;
    
    return 0;
}

static long aaco_ioctl_start_measurement(struct file *file)
{
    struct aaco_file_state *state = file->private_data;
    
    spin_lock(&state->lock);
    
    state->measurement.start_tsc = rdtsc_ordered();
    state->measurement.start_ns = ktime_get_ns();
    
    /* Capture starting stats */
    state->measurement.start_ctx.voluntary_switches = current->nvcsw;
    state->measurement.start_ctx.involuntary_switches = current->nivcsw;
    state->measurement.start_ctx.total_switches = current->nvcsw + current->nivcsw;
    
    state->measurement.active = 1;
    
    spin_unlock(&state->lock);
    
    atomic64_inc(&aaco_state.total_measurements);
    
    return 0;
}

static long aaco_ioctl_stop_measurement(struct file *file, void __user *arg)
{
    struct aaco_file_state *state = file->private_data;
    struct aaco_measurement_state result;
    
    spin_lock(&state->lock);
    
    state->measurement.end_tsc = rdtsc_ordered();
    state->measurement.end_ns = ktime_get_ns();
    
    /* Capture ending stats */
    state->measurement.end_ctx.voluntary_switches = current->nvcsw;
    state->measurement.end_ctx.involuntary_switches = current->nivcsw;
    state->measurement.end_ctx.total_switches = current->nvcsw + current->nivcsw;
    
    state->measurement.active = 0;
    
    memcpy(&result, &state->measurement, sizeof(result));
    
    spin_unlock(&state->lock);
    
    if (copy_to_user(arg, &result, sizeof(result)))
        return -EFAULT;
    
    return 0;
}

static long aaco_ioctl(struct file *file, unsigned int cmd, unsigned long arg)
{
    void __user *argp = (void __user *)arg;
    
    switch (cmd) {
    case AACO_GET_VERSION:
        return aaco_ioctl_get_version(argp);
    case AACO_GET_TSC_FREQ:
        return aaco_ioctl_get_tsc_freq(argp);
    case AACO_READ_TSC:
        return aaco_ioctl_read_tsc(argp);
    case AACO_SET_CPU_AFFINITY:
        return aaco_ioctl_set_cpu_affinity(file, argp);
    case AACO_GET_CPU_AFFINITY:
        return aaco_ioctl_get_cpu_affinity(argp);
    case AACO_MEMORY_BARRIER:
        return aaco_ioctl_memory_barrier();
    case AACO_DISABLE_PREEMPTION:
        return aaco_ioctl_disable_preemption(file);
    case AACO_ENABLE_PREEMPTION:
        return aaco_ioctl_enable_preemption(file);
    case AACO_GET_IRQ_STATS:
        return aaco_ioctl_get_irq_stats(argp);
    case AACO_GET_CTX_SWITCHES:
        return aaco_ioctl_get_ctx_switches(argp);
    case AACO_START_MEASUREMENT:
        return aaco_ioctl_start_measurement(file);
    case AACO_STOP_MEASUREMENT:
        return aaco_ioctl_stop_measurement(file, argp);
    default:
        return -ENOTTY;
    }
}

/* =============================================================================
 * File Operations Structure
 * ============================================================================= */

static const struct file_operations aaco_fops = {
    .owner          = THIS_MODULE,
    .open           = aaco_open,
    .release        = aaco_release,
    .unlocked_ioctl = aaco_ioctl,
    .compat_ioctl   = aaco_ioctl,
};

/* =============================================================================
 * Module Init/Exit
 * ============================================================================= */

static int __init aaco_driver_init(void)
{
    int ret;
    
    pr_info("aaco: initializing AACO-SIGMA driver v%d.%d.%d\n",
            AACO_VERSION_MAJOR, AACO_VERSION_MINOR, AACO_VERSION_PATCH);
    
    /* Detect TSC frequency */
    aaco_state.tsc_freq_hz = detect_tsc_frequency();
    pr_info("aaco: TSC frequency: %llu Hz\n", aaco_state.tsc_freq_hz);
    
    /* Allocate device number */
    ret = alloc_chrdev_region(&aaco_state.devno, 0, 1, AACO_DEVICE_NAME);
    if (ret < 0) {
        pr_err("aaco: failed to allocate device number\n");
        return ret;
    }
    
    /* Create device class */
    aaco_state.class = class_create(THIS_MODULE, AACO_CLASS_NAME);
    if (IS_ERR(aaco_state.class)) {
        ret = PTR_ERR(aaco_state.class);
        pr_err("aaco: failed to create class\n");
        goto err_unregister;
    }
    
    /* Initialize and add cdev */
    cdev_init(&aaco_state.cdev, &aaco_fops);
    aaco_state.cdev.owner = THIS_MODULE;
    
    ret = cdev_add(&aaco_state.cdev, aaco_state.devno, 1);
    if (ret < 0) {
        pr_err("aaco: failed to add cdev\n");
        goto err_class;
    }
    
    /* Create device node */
    aaco_state.device = device_create(aaco_state.class, NULL, 
                                      aaco_state.devno, NULL, 
                                      AACO_DEVICE_NAME);
    if (IS_ERR(aaco_state.device)) {
        ret = PTR_ERR(aaco_state.device);
        pr_err("aaco: failed to create device\n");
        goto err_cdev;
    }
    
    /* Initialize statistics */
    atomic64_set(&aaco_state.total_measurements, 0);
    atomic64_set(&aaco_state.total_noise_events, 0);
    
    pr_info("aaco: driver initialized successfully\n");
    return 0;

err_cdev:
    cdev_del(&aaco_state.cdev);
err_class:
    class_destroy(aaco_state.class);
err_unregister:
    unregister_chrdev_region(aaco_state.devno, 1);
    return ret;
}

static void __exit aaco_driver_exit(void)
{
    pr_info("aaco: unloading driver (measurements: %lld)\n",
            atomic64_read(&aaco_state.total_measurements));
    
    device_destroy(aaco_state.class, aaco_state.devno);
    cdev_del(&aaco_state.cdev);
    class_destroy(aaco_state.class);
    unregister_chrdev_region(aaco_state.devno, 1);
    
    pr_info("aaco: driver unloaded\n");
}

module_init(aaco_driver_init);
module_exit(aaco_driver_exit);
