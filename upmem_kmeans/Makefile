# Define the compiler and flags
DPU_CC = dpu-upmem-dpurte-clang
HOST_CC = gcc
CFLAGS = -DNR_TASKLETS=4
HOST_CFLAGS = --std=c99 -g
LDFLAGS = `dpu-pkg-config --cflags --libs dpu` -lm

# Define the source files and targets
DPU_SRCS = avg_coordinate.c distance_matrix.c
HOST_SRCS = kmeans.c
DPU_TARGETS = avg_coordinate distance_matrix
HOST_TARGET = kmeans

# Default target
all: $(DPU_TARGETS) $(HOST_TARGET)

# Compile DPU programs
avg_coordinate: avg_coordinate.c
	$(DPU_CC) $(CFLAGS) $< -o $@

distance_matrix: distance_matrix.c
	$(DPU_CC) $(CFLAGS) $< -o $@

# Compile host program
kmeans: kmeans.c
	$(HOST_CC) $(HOST_CFLAGS) $< -o $@ $(LDFLAGS)

# Clean up
clean:
	rm -f $(DPU_TARGETS) $(HOST_TARGET)