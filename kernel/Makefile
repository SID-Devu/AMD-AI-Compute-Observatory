# AACO Linux Kernel Module Makefile
# Build the driver-integrated observability module

obj-m := aaco.o
aaco-objs := aaco_driver.o

KDIR ?= /lib/modules/$(shell uname -r)/build
PWD := $(shell pwd)

# Compiler flags
ccflags-y := -DDEBUG -g -Wall -Werror

.PHONY: all clean install uninstall load unload

all:
	$(MAKE) -C $(KDIR) M=$(PWD) modules

clean:
	$(MAKE) -C $(KDIR) M=$(PWD) clean
	rm -f *.o *.ko *.mod.c *.mod.o .*.cmd modules.order Module.symvers

install: all
	sudo cp aaco.ko /lib/modules/$(shell uname -r)/kernel/drivers/misc/
	sudo depmod -a
	@echo "AACO driver installed. Load with: sudo modprobe aaco"

uninstall:
	sudo rm -f /lib/modules/$(shell uname -r)/kernel/drivers/misc/aaco.ko
	sudo depmod -a
	@echo "AACO driver uninstalled"

load:
	sudo insmod aaco.ko
	@echo "AACO driver loaded"
	@echo "Check /dev/aaco and /sys/kernel/debug/aaco/"

unload:
	sudo rmmod aaco
	@echo "AACO driver unloaded"

# Create device node (if not using udev)
mknod:
	sudo mknod /dev/aaco c $(shell grep aaco /proc/devices | cut -d' ' -f1) 0
	sudo chmod 666 /dev/aaco

# Show driver info
info:
	@echo "=== AACO Driver Status ==="
	@lsmod | grep aaco || echo "Driver not loaded"
	@echo ""
	@ls -la /dev/aaco 2>/dev/null || echo "/dev/aaco not found"
	@echo ""
	@cat /sys/kernel/debug/aaco/stats 2>/dev/null || echo "debugfs not available"

# Run tests
test: load
	@echo "Running basic driver test..."
	@python3 ../tests/test_driver.py
