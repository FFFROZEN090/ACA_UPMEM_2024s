#include <dpu.h>
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <math.h>

#ifndef DPU_BINARY
#define DPU_BINARY "distance_matrix"
#endif


/* Number of points */
#define TOTAL_NUM_POINTS 4096 // Example for 64 points
#define NUM_POINTS 8
#define DPU_NUMBER 4

/* NxN matrix */
#define DISTANCE_VECTOR_SIZE TOTAL_NUM_POINTS
#define DISTANCE_MATRIX_SIZE TOTAL_NUM_POINTS * TOTAL_NUM_POINTS


// Populate the data to the DPUs
void populate_mram(struct dpu_set_t set, struct dpu_set_t dpu, uint8_t *points) {

    uint32_t num_points_per_dpu = TOTAL_NUM_POINTS / DPU_NUMBER;
    printf("Points per DPU: %d\n", num_points_per_dpu);
    uint32_t each_dpu;

    DPU_FOREACH(set, dpu, each_dpu){
        // print dpu id
        printf("DPU ID: %d\n", each_dpu);
        // Reconstruc the points for each DPU
        DPU_ASSERT(dpu_prepare_xfer(dpu, &points[each_dpu * num_points_per_dpu * 2]));
        
    }
    DPU_ASSERT(dpu_push_xfer(set, DPU_XFER_TO_DPU, "points", 0, num_points_per_dpu * 2 * sizeof(uint8_t), DPU_XFER_DEFAULT));

    // Populate the end of the points to each DPU, because one DPU will handle 1024 points
    DPU_FOREACH(set, dpu, each_dpu){
        // Reconstruc the points for each DPU
        DPU_ASSERT(dpu_copy_to(dpu, "total_num", 0, &num_points_per_dpu, sizeof(uint32_t)));
        
    }
    

}

// Populate the points to the DPUs for average coordinate calculation
void populate_mram_avg(struct dpu_set_t set, struct dpu_set_t dpu, uint8_t *points) {
    uint32_t num_points_per_dpu = TOTAL_NUM_POINTS / DPU_NUMBER;
    uint32_t each_dpu;

    DPU_FOREACH(set, dpu, each_dpu){
        // print dpu id
        printf("DPU ID: %d\n", each_dpu);
        // Reconstruc the points for each DPU
        DPU_ASSERT(dpu_prepare_xfer(dpu, &points[each_dpu * num_points_per_dpu * 2]));
        
    }
    DPU_ASSERT(dpu_push_xfer(set, DPU_XFER_TO_DPU, "points", 0, num_points_per_dpu * 2 * sizeof(uint8_t), DPU_XFER_DEFAULT));
}

// Generate the coordinates of the points, the axis is uint8_t data type
void generate_points(uint8_t *points) {
    for (int i = 0; i < TOTAL_NUM_POINTS * 2 - 1; i+=2) {
        // Assign random values to the points
        points[i] = rand() % 256;
        points[i + 1] = rand() % 256;
    }
}



int main() {
    struct dpu_set_t set, dpu;
    uint64_t distance[DISTANCE_VECTOR_SIZE];

    // Initial total points
    uint8_t points[TOTAL_NUM_POINTS * 2];

    // Generate the points
    generate_points(points);

    // printf the first 10 points
    for (int i = 0; i < 19; i+=2) {
        printf("Point %d: (%d, %d)\n", i, points[i], points[i + 1]);
    }

    DPU_ASSERT(dpu_alloc(DPU_NUMBER, NULL, &set));

    DPU_ASSERT(dpu_load(set, DPU_BINARY, NULL));

    // Load data to the DPUs
    populate_mram(set, dpu, points);

    // Calculate how many points each DPU will handle
    int num_points_per_dpu = TOTAL_NUM_POINTS / DPU_NUMBER;

    // Launch the DPUs
    DPU_ASSERT(dpu_launch(set, DPU_SYNCHRONOUS));

    // Init each DPU variables
    uint32_t each_dpu;
    // Copy the result back

    DPU_FOREACH(set, dpu, each_dpu){
        DPU_ASSERT(dpu_prepare_xfer(dpu, &distance[each_dpu * num_points_per_dpu]));
    }

    DPU_ASSERT(dpu_push_xfer(set, DPU_XFER_FROM_DPU, "distance", 0, num_points_per_dpu * sizeof(uint64_t), DPU_XFER_DEFAULT));

    // Print the first 10 distances
    for (int i = 0; i < 9; i++) {
        printf("Distance %d: %ld\n", i, distance[i]);
    }
    

    DPU_ASSERT(dpu_free(set));

    // Start the kernel of avg_coordinate.c

    // Initialize the DPUs
    DPU_ASSERT(dpu_alloc(DPU_NUMBER, NULL, &set));

    // Load the binary
    DPU_ASSERT(dpu_load(set, "avg_coordinate", NULL));

    // Load the data to the DPUs
    populate_mram_avg(set, dpu, points);

    // Launch the DPUs
    DPU_ASSERT(dpu_launch(set, DPU_SYNCHRONOUS));

    // Initial the final index list
    uint64_t final_min_index[DPU_NUMBER];

    // Copy the result back
    DPU_FOREACH(set, dpu, each_dpu){
        DPU_ASSERT(dpu_copy_from(dpu, "final_min_index", 0, &final_min_index[each_dpu], sizeof(uint64_t)));
    }


    // Print the final index
    for (int i = 0; i < DPU_NUMBER; i++) {
        printf("Final index %d: %ld\n", i, final_min_index[i]);
    }

    // Free the DPUs
    DPU_ASSERT(dpu_free(set));


    return 0;
}
