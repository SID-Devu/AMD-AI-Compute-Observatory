# AACO eBPF Programs Makefile
# Compiles eBPF programs for GPU profiling

CLANG ?= clang
LLVM_STRIP ?= llvm-strip
BPFTOOL ?= bpftool

# Kernel headers
KERNEL_HEADERS ?= /usr/src/linux-headers-$(shell uname -r)

# Compiler flags
CFLAGS := -g -O2 -target bpf
CFLAGS += -D__TARGET_ARCH_x86
CFLAGS += -I$(KERNEL_HEADERS)/include
CFLAGS += -I$(KERNEL_HEADERS)/include/uapi
CFLAGS += -I$(KERNEL_HEADERS)/arch/x86/include
CFLAGS += -I$(KERNEL_HEADERS)/arch/x86/include/uapi
CFLAGS += -I/usr/include/bpf

# Source files
SRCS := $(wildcard *.bpf.c)
OBJS := $(SRCS:.bpf.c=.bpf.o)

# Default target
all: $(OBJS)

# Compile BPF programs
%.bpf.o: %.bpf.c
	$(CLANG) $(CFLAGS) -c $< -o $@
	$(LLVM_STRIP) -g $@

# Generate skeleton headers (optional, for embedding)
%.skel.h: %.bpf.o
	$(BPFTOOL) gen skeleton $< > $@

# Clean build artifacts
clean:
	rm -f *.o *.skel.h

# Install to system location
install: $(OBJS)
	install -d /usr/share/aaco/ebpf
	install -m 644 $(OBJS) /usr/share/aaco/ebpf/

# Check if we can build
check:
	@echo "Checking build environment..."
	@which $(CLANG) > /dev/null || (echo "Error: clang not found" && exit 1)
	@echo "Clang: $(shell $(CLANG) --version | head -1)"
	@test -d $(KERNEL_HEADERS) || (echo "Error: Kernel headers not found at $(KERNEL_HEADERS)" && exit 1)
	@echo "Kernel headers: $(KERNEL_HEADERS)"
	@echo "Build environment OK"

.PHONY: all clean install check
