# AACO-SIGMA Linux Kernel Driver Makefile
#
# Build: make
# Install: sudo make install
# Clean: make clean
#
# Requires kernel headers for the running kernel

obj-m := aaco_driver.o

KDIR ?= /lib/modules/$(shell uname -r)/build
PWD := $(shell pwd)

ccflags-y := -DDEBUG

all: modules

modules:
	$(MAKE) -C $(KDIR) M=$(PWD) modules

clean:
	$(MAKE) -C $(KDIR) M=$(PWD) clean

install: modules
	sudo insmod aaco_driver.ko

uninstall:
	-sudo rmmod aaco_driver

# Install to system modules directory
install-system: modules
	sudo cp aaco_driver.ko /lib/modules/$(shell uname -r)/extra/
	sudo depmod -a

# Create udev rule for device permissions
install-udev:
	@echo 'KERNEL=="aaco", MODE="0666"' | sudo tee /etc/udev/rules.d/99-aaco.rules
	sudo udevadm control --reload-rules

# Full installation
install-full: install-system install-udev
	sudo modprobe aaco_driver

# Development: load module with debugging
dev: modules
	-sudo rmmod aaco_driver 2>/dev/null
	sudo insmod aaco_driver.ko
	dmesg | tail -20

# Check if module is loaded
status:
	@lsmod | grep aaco || echo "Module not loaded"
	@ls -la /dev/aaco 2>/dev/null || echo "Device not found"

.PHONY: all modules clean install uninstall install-system install-udev install-full dev status
