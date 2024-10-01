#include <defs.h>
#include <mram.h>
#include <perfcounter.h>
#include <stdint.h>
#include <stdio.h>
#include <barrier.h>
#include <vmutex.h>

// Total number of points
#define TOTAL_NUM_POINTS 1024
#define NR_TASKLETS 4

__mram_noinit uint8_t points[TOTAL_NUM_POINTS * 2];
__mram uint64_t distance[TOTAL_NUM_POINTS];
__dma_aligned uint64_t x_sum[NR_TASKLETS];
__dma_aligned uint64_t y_sum[NR_TASKLETS];
__host uint64_t total[2];

// Barrier for synchronization
BARRIER_INIT(my_barrier, NR_TASKLETS);

// Function to sum x and y values
void sum_xy_values(int A, uint64_t *x_sum_tasklet, uint64_t *y_sum_tasklet) {
    *x_sum_tasklet += points[A];
    *y_sum_tasklet += points[A + 1];
}

// Find the closest point to the total average in each tasklet then return the distance and index
void find_closest_point(uint64_t total_x_sum, uint64_t total_y_sum, uint64_t *min_distance, uint64_t *min_index, uint32_t tasklet_id) {
    uint64_t min_dist = UINT64_MAX;
    uint64_t min_idx = 0;

    // Calculate the number of points per tasklet
    int num_points_per_tasklet = TOTAL_NUM_POINTS / NR_TASKLETS;
    for (int i = tasklet_id * num_points_per_tasklet * 2; i < (tasklet_id + 1) * num_points_per_tasklet * 2; i += 2) {
        uint64_t dx = total_x_sum - points[i];
        uint64_t dy = total_y_sum - points[i + 1];
        uint64_t dist = dx * dx + dy * dy;
        if (dist < min_dist) {
            min_dist = dist;
            min_idx = i / 2;
        }
    }

    // Store the result in the global arrays
    min_distance[tasklet_id] = min_dist;
    min_index[tasklet_id] = min_idx;
}

int main() {
    // Initialize performance counter
    perfcounter_config(COUNT_CYCLES, true);

    // Get tasklet ID
    uint32_t tasklet_id = me();

    // Each Tasklet handles TOTAL_NUM_POINTS/NR_TASKLETS points
    int num_points_per_tasklet = TOTAL_NUM_POINTS / NR_TASKLETS;

    // Initialize sum variables for this tasklet
    __dma_aligned uint64_t local_x_sum = 0;
    __dma_aligned uint64_t local_y_sum = 0;

    // Sum the x and y values
    for (int i = tasklet_id * num_points_per_tasklet * 2; i < (tasklet_id + 1) * num_points_per_tasklet * 2; i += 2) {
        sum_xy_values(i, &local_x_sum, &local_y_sum);
    }

    // Store the result in the global arrays
    x_sum[tasklet_id] = local_x_sum;
    y_sum[tasklet_id] = local_y_sum;

    // Barrier to ensure all tasklets have finished calculating
    barrier_wait(&my_barrier);

    // Tasklet 0 aggregates the results
    if (tasklet_id == 0) {

        for (int i = 0; i < NR_TASKLETS; i++) {
            total[0] += x_sum[i];
            total[1] += y_sum[i];
        }
    }

    // Print the total x and y sum
    if (tasklet_id == 0) {
        printf("Total x sum: %lu\n", total[0]);
        printf("Total y sum: %lu\n", total[1]);
    }

    // Barrier to ensure all tasklets have finished aggregating
    barrier_wait(&my_barrier);


    return 0;
}
