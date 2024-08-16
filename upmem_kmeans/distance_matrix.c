#include <defs.h>
#include <mram.h>
#include <perfcounter.h>
#include <stdint.h>
#include <stdio.h>
#include <barrier.h>
#include <vmutex.h>
//Total number of points
// How many points for one WRAM buffer
#define TOTAL_NUM_POINTS 1024
#define NR_TASKLETS 4


__mram_noinit uint8_t points[TOTAL_NUM_POINTS*2];
__mram uint64_t distance[TOTAL_NUM_POINTS];

// Barrier for synchronization
BARRIER_INIT(my_barrier, NR_TASKLETS);


// Calculate the distance matrix
uint64_t calculate_distance(int A) {
    uint64_t dx = points[A] - points[0];
    uint64_t dy = points[A + 1] - points[1];
    return (dx * dx + dy * dy);
}

int main() {
    // Get tasklet ID
    uint32_t tasklet_id = me();

    // Each Tasklet is to handle TOTAL_NUM_POINTS/NUM_POINTS points
    int num_points_per_tasklet = TOTAL_NUM_POINTS / NR_TASKLETS;

    // Calculate the distance matrix
    for (int i = tasklet_id * num_points_per_tasklet * 2; i < (tasklet_id + 1) * num_points_per_tasklet * 2 ; i+=2) {;
        uint64_t dis = calculate_distance(i);
        distance[i/2] = dis;  
    }

    // Synchronize all tasklets
    barrier_wait(&my_barrier);

    // For the first tasklet, print the distance matrix
    if (tasklet_id == 0) {
        for (int i = 0; i < TOTAL_NUM_POINTS; i++) {
            printf("Distance[%d]: %lu", i, distance[i]);
        }
    }
    return 0;
}
