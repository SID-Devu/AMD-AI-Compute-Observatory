/* SPDX-License-Identifier: GPL-2.0 */
/*
 * AACO Kernel Module
 * 
 * Provides low-level GPU profiling capabilities via kernel interfaces.
 * Exposes performance counters and timing data to userspace.
 */

#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/fs.h>
#include <linux/device.h>
#include <linux/cdev.h>
#include <linux/uaccess.h>
#include <linux/slab.h>
#include <linux/mutex.h>
#include <linux/list.h>
#include <linux/spinlock.h>

#define AACO_KMOD_NAME    "aaco"
#define AACO_KMOD_VERSION "1.0.0"

/* IOCTL commands */
#define AACO_IOC_MAGIC    'A'
#define AACO_IOC_START    _IO(AACO_IOC_MAGIC, 0)
#define AACO_IOC_STOP     _IO(AACO_IOC_MAGIC, 1)
#define AACO_IOC_READ     _IOR(AACO_IOC_MAGIC, 2, struct aaco_sample)
#define AACO_IOC_CONFIG   _IOW(AACO_IOC_MAGIC, 3, struct aaco_config)

/* Sample data structure */
struct aaco_sample {
    __u64 timestamp;
    __u32 gpu_id;
    __u32 type;
    __u64 value;
    __u64 metadata;
};

/* Configuration structure */
struct aaco_config {
    __u32 sample_interval_us;
    __u32 gpu_mask;
    __u32 counter_mask;
    __u32 flags;
};

/* Module state */
static int aaco_major;
static struct class *aaco_class;
static struct cdev aaco_cdev;
static struct device *aaco_device;

static DEFINE_MUTEX(aaco_mutex);
static bool aaco_active;
static struct aaco_config current_config;

/* Sample buffer */
#define SAMPLE_BUFFER_SIZE 4096
static struct aaco_sample *sample_buffer;
static unsigned int sample_head;
static unsigned int sample_tail;
static DEFINE_SPINLOCK(sample_lock);

/* Forward declarations */
static int aaco_open(struct inode *inode, struct file *file);
static int aaco_release(struct inode *inode, struct file *file);
static ssize_t aaco_read(struct file *file, char __user *buf, 
                         size_t count, loff_t *pos);
static long aaco_ioctl(struct file *file, unsigned int cmd, unsigned long arg);

/* File operations */
static const struct file_operations aaco_fops = {
    .owner          = THIS_MODULE,
    .open           = aaco_open,
    .release        = aaco_release,
    .read           = aaco_read,
    .unlocked_ioctl = aaco_ioctl,
};

static int aaco_open(struct inode *inode, struct file *file)
{
    if (!mutex_trylock(&aaco_mutex))
        return -EBUSY;
    
    pr_info("aaco: device opened\n");
    return 0;
}

static int aaco_release(struct inode *inode, struct file *file)
{
    mutex_unlock(&aaco_mutex);
    pr_info("aaco: device closed\n");
    return 0;
}

static ssize_t aaco_read(struct file *file, char __user *buf,
                         size_t count, loff_t *pos)
{
    struct aaco_sample sample;
    unsigned long flags;
    ssize_t ret = 0;
    
    if (count < sizeof(sample))
        return -EINVAL;
    
    spin_lock_irqsave(&sample_lock, flags);
    
    if (sample_head != sample_tail) {
        sample = sample_buffer[sample_tail];
        sample_tail = (sample_tail + 1) % SAMPLE_BUFFER_SIZE;
        ret = sizeof(sample);
    }
    
    spin_unlock_irqrestore(&sample_lock, flags);
    
    if (ret > 0) {
        if (copy_to_user(buf, &sample, sizeof(sample)))
            return -EFAULT;
    }
    
    return ret;
}

static long aaco_ioctl(struct file *file, unsigned int cmd, unsigned long arg)
{
    struct aaco_config config;
    
    switch (cmd) {
    case AACO_IOC_START:
        aaco_active = true;
        pr_info("aaco: profiling started\n");
        return 0;
        
    case AACO_IOC_STOP:
        aaco_active = false;
        pr_info("aaco: profiling stopped\n");
        return 0;
        
    case AACO_IOC_CONFIG:
        if (copy_from_user(&config, (void __user *)arg, sizeof(config)))
            return -EFAULT;
        
        current_config = config;
        pr_info("aaco: config updated (interval=%u us)\n",
                config.sample_interval_us);
        return 0;
        
    default:
        return -ENOTTY;
    }
}

/* Add sample to buffer */
void aaco_add_sample(struct aaco_sample *sample)
{
    unsigned long flags;
    unsigned int next_head;
    
    if (!aaco_active)
        return;
    
    spin_lock_irqsave(&sample_lock, flags);
    
    next_head = (sample_head + 1) % SAMPLE_BUFFER_SIZE;
    
    if (next_head != sample_tail) {
        sample_buffer[sample_head] = *sample;
        sample_head = next_head;
    }
    
    spin_unlock_irqrestore(&sample_lock, flags);
}
EXPORT_SYMBOL(aaco_add_sample);

/* Module init */
static int __init aaco_init(void)
{
    int ret;
    dev_t dev;
    
    pr_info("aaco: initializing module v%s\n", AACO_KMOD_VERSION);
    
    /* Allocate sample buffer */
    sample_buffer = kmalloc_array(SAMPLE_BUFFER_SIZE, 
                                  sizeof(struct aaco_sample),
                                  GFP_KERNEL);
    if (!sample_buffer) {
        pr_err("aaco: failed to allocate sample buffer\n");
        return -ENOMEM;
    }
    
    /* Allocate device number */
    ret = alloc_chrdev_region(&dev, 0, 1, AACO_KMOD_NAME);
    if (ret < 0) {
        pr_err("aaco: failed to allocate device number\n");
        goto err_free_buffer;
    }
    aaco_major = MAJOR(dev);
    
    /* Create device class */
    aaco_class = class_create(THIS_MODULE, AACO_KMOD_NAME);
    if (IS_ERR(aaco_class)) {
        ret = PTR_ERR(aaco_class);
        pr_err("aaco: failed to create class\n");
        goto err_unregister;
    }
    
    /* Initialize cdev */
    cdev_init(&aaco_cdev, &aaco_fops);
    aaco_cdev.owner = THIS_MODULE;
    
    ret = cdev_add(&aaco_cdev, dev, 1);
    if (ret < 0) {
        pr_err("aaco: failed to add cdev\n");
        goto err_destroy_class;
    }
    
    /* Create device */
    aaco_device = device_create(aaco_class, NULL, dev, NULL, AACO_KMOD_NAME);
    if (IS_ERR(aaco_device)) {
        ret = PTR_ERR(aaco_device);
        pr_err("aaco: failed to create device\n");
        goto err_del_cdev;
    }
    
    pr_info("aaco: module loaded (major=%d)\n", aaco_major);
    return 0;

err_del_cdev:
    cdev_del(&aaco_cdev);
err_destroy_class:
    class_destroy(aaco_class);
err_unregister:
    unregister_chrdev_region(dev, 1);
err_free_buffer:
    kfree(sample_buffer);
    return ret;
}

/* Module exit */
static void __exit aaco_exit(void)
{
    device_destroy(aaco_class, MKDEV(aaco_major, 0));
    cdev_del(&aaco_cdev);
    class_destroy(aaco_class);
    unregister_chrdev_region(MKDEV(aaco_major, 0), 1);
    kfree(sample_buffer);
    
    pr_info("aaco: module unloaded\n");
}

module_init(aaco_init);
module_exit(aaco_exit);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("AACO Team");
MODULE_DESCRIPTION("AACO GPU Profiling Kernel Module");
MODULE_VERSION(AACO_KMOD_VERSION);
