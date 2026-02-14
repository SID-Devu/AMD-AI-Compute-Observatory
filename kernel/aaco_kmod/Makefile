# AACO Kernel Module Makefile

obj-m := aaco_kmod.o

# Kernel build directory
KDIR ?= /lib/modules/$(shell uname -r)/build

# Module name
MODULE_NAME := aaco_kmod

# Extra compiler flags
ccflags-y := -Wall -Werror

# Default target
all:
	$(MAKE) -C $(KDIR) M=$(PWD) modules

# Clean target
clean:
	$(MAKE) -C $(KDIR) M=$(PWD) clean
	rm -f Module.symvers modules.order

# Install module
install: all
	$(MAKE) -C $(KDIR) M=$(PWD) modules_install
	depmod -a

# Uninstall module
uninstall:
	rm -f /lib/modules/$(shell uname -r)/extra/$(MODULE_NAME).ko
	depmod -a

# Load module
load:
	modprobe $(MODULE_NAME) || insmod $(MODULE_NAME).ko

# Unload module
unload:
	rmmod $(MODULE_NAME) || true

# Reload module
reload: unload load

# Create device node (if not using udev)
device:
	@if [ ! -c /dev/aaco ]; then \
		MAJOR=$$(grep aaco /proc/devices | awk '{print $$1}'); \
		if [ -n "$$MAJOR" ]; then \
			mknod /dev/aaco c $$MAJOR 0; \
			chmod 666 /dev/aaco; \
			echo "Created /dev/aaco with major $$MAJOR"; \
		else \
			echo "Module not loaded"; \
		fi \
	else \
		echo "/dev/aaco already exists"; \
	fi

# Check module status
status:
	@lsmod | grep $(MODULE_NAME) || echo "Module not loaded"
	@if [ -c /dev/aaco ]; then \
		ls -la /dev/aaco; \
	else \
		echo "Device /dev/aaco not found"; \
	fi

# View kernel log for AACO
log:
	dmesg | grep aaco | tail -20

.PHONY: all clean install uninstall load unload reload device status log
