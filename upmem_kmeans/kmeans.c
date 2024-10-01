#include <dpu.h>
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <math.h>
// Record execution time
#include <time.h>

#ifndef DISTANCE_MATRIX
#define DISTANCE_MATRIX "distance_matrix"
#endif

#ifndef AVG_COORDINATE
#define AVG_COORDINATE "avg_coordinate"
#endif


/* Number of points */
#define TOTAL_NUM_POINTS 4092 // Example for 64 points

/* Number of centroids */
#define NUM_CENTROIDS 4

#define DPU_NUMBER 4

/* NxN matrix */
#define DISTANCE_MATRIX_SIZE NUM_CENTROIDS * (TOTAL_NUM_POINTS + 1)


/* Populate the data to the DPUs for distance matrix calculation 
    1. Calculate how many points each DPU will handle denote as num_points_per_dpu
    2. Prepare a [number of points + 1] array for each DPU, the first element is the centroid
    3. Populate the data to the DPUs
*/
void populate_mram(struct dpu_set_t set, struct dpu_set_t dpu, uint8_t *points, uint16_t centroid_index) {

    uint16_t num_points_per_dpu = TOTAL_NUM_POINTS / DPU_NUMBER;
    // Check if the number of points is exceed the limit with 1023 points per DPU
    if (num_points_per_dpu > 1023) {
        printf("The number of points per DPU is exceed the limit\n");
        return;
    }

    // Assign a new points array for DPUs, which evenly insert the centroids before each num_points_per_dpu points
    uint8_t new_points[(num_points_per_dpu + 1) * DPU_NUMBER * 2];

    // Assign the current centroid to the first element of each DPU
    for (int i = 0; i < DPU_NUMBER; i++) {
        new_points[i * (num_points_per_dpu + 1) * 2] = points[centroid_index * 2];
        new_points[i * (num_points_per_dpu + 1) * 2 + 1] = points[centroid_index * 2 + 1];
    }

    // Assign the rest of the points to the new points array
    for (int i = 0; i < DPU_NUMBER; i++) {
        for (int j = 0; j < num_points_per_dpu * 2; j++) {
            new_points[i * (num_points_per_dpu + 1) * 2 + 2 + j] = points[i * num_points_per_dpu * 2 + j];
        }
    }

    uint32_t each_dpu;

    DPU_FOREACH(set, dpu, each_dpu){

        // Prepare the data for each DPU
        DPU_ASSERT(dpu_prepare_xfer(dpu, &new_points[each_dpu * (num_points_per_dpu + 1) * 2]));
    }
    DPU_ASSERT(dpu_push_xfer(set, DPU_XFER_TO_DPU, "points", 0, (num_points_per_dpu + 1) * 2 * sizeof(uint8_t), DPU_XFER_DEFAULT));
}

// Populate the points to the DPUs for average coordinate calculation
void populate_mram_avg(struct dpu_set_t set, struct dpu_set_t dpu, uint8_t *points) {
    uint32_t each_dpu;

    DPU_FOREACH(set, dpu, each_dpu){
        // Prepare the data for each DPU
        DPU_ASSERT(dpu_prepare_xfer(dpu, &points[each_dpu * 1024 * 2]));
    }
    DPU_ASSERT(dpu_push_xfer(set, DPU_XFER_TO_DPU, "points", 0, 1024 * 2 * sizeof(uint8_t), DPU_XFER_DEFAULT));
}

/* 
    From distance matrix find nearest points to the centroids
        1. Loop through all the points
        2. Find the nearest centroid to the point
        3. For each point, record the the nearest centroid index
*/ 
void find_nearest_centroid(uint64_t *distance_matrix, uint16_t *nearest_centroid, uint16_t *centroids) {
    for (int i = 0; i < TOTAL_NUM_POINTS; i++) {
        // Every first element is the centroid
        uint64_t min_distance = distance_matrix[i];
        uint16_t min_centroid = 0;
        for (int j = 1; j < NUM_CENTROIDS; j++) {
            if (distance_matrix[i * NUM_CENTROIDS + j] < min_distance) {
                min_distance = distance_matrix[i * NUM_CENTROIDS + j];
                min_centroid = j;
            }
        }
        nearest_centroid[i] = min_centroid;
    }
}

// CPU version calculate the distance matrix
void calculate_distance_matrix(uint8_t *points, uint64_t *distance_matrix, uint16_t *centroids) {
    for (int i = 0; i < NUM_CENTROIDS; i++) {
        for (int j = 0; j < TOTAL_NUM_POINTS; j++) {
            // Calculate the distance between the centroid and the points
            distance_matrix[i * (TOTAL_NUM_POINTS + 1) + j] = pow(points[centroids[i] * 2] - points[j * 2], 2) + pow(points[centroids[i] * 2 + 1] - points[j * 2 + 1], 2);
        }
    }
}

/* 
    Populate the data to the DPUs for average coordinate calculation
        Input:
            set: the DPU set
            dpu: the DPU
            points: the coordinates of the points
            nearest_centroid: the nearest centroid to each point
            x_sum: the sum of x coordinates for each centroid
            y_sum: the sum of y coordinates for each centroid
*/
void calculate_avg_coordinate(struct dpu_set_t set, struct dpu_set_t dpu, uint8_t *points, uint16_t *nearest_centroid, uint64_t *x_sum, uint64_t *y_sum) {
    uint32_t num_points_per_dpu = TOTAL_NUM_POINTS / DPU_NUMBER;
    uint32_t each_dpu;

    DPU_FOREACH(set, dpu, each_dpu){
        // Prepare the data for each DPU
        DPU_ASSERT(dpu_prepare_xfer(dpu, &points[each_dpu * num_points_per_dpu * 2]));
        DPU_ASSERT(dpu_prepare_xfer(dpu, &nearest_centroid[each_dpu * num_points_per_dpu]));
    }
    DPU_ASSERT(dpu_push_xfer(set, DPU_XFER_TO_DPU, "points", 0, num_points_per_dpu * 2 * sizeof(uint8_t), DPU_XFER_DEFAULT));
    DPU_ASSERT(dpu_push_xfer(set, DPU_XFER_TO_DPU, "nearest_centroid", 0, num_points_per_dpu * sizeof(uint16_t), DPU_XFER_DEFAULT));

    // Execute the DPU program
    DPU_ASSERT(dpu_launch(set, DPU_SYNCHRONOUS));

    // Get the result from the DPUs
    DPU_FOREACH(set, dpu, each_dpu){
        // Prepare the data for each DPU
        DPU_ASSERT(dpu_prepare_xfer(dpu, &x_sum[each_dpu]));
        DPU_ASSERT(dpu_prepare_xfer(dpu, &y_sum[each_dpu]));
    }
    DPU_ASSERT(dpu_push_xfer(set, DPU_XFER_FROM_DPU, "x_sum", 0, num_points_per_dpu * sizeof(uint64_t), DPU_XFER_DEFAULT));
    DPU_ASSERT(dpu_push_xfer(set, DPU_XFER_FROM_DPU, "y_sum", 0, num_points_per_dpu * sizeof(uint64_t), DPU_XFER_DEFAULT));
}


// Generate the coordinates of the points, the axis is uint8_t data type
void generate_points(uint8_t *points) {
    for (int i = 0; i < TOTAL_NUM_POINTS * 2 - 1; i+=2) {
        // Assign random values to the points
        points[i] = rand() % 255;
        points[i + 1] = rand() % 255;
    }
}



int main() {
    // Initialize the dataset
    uint8_t points[TOTAL_NUM_POINTS * 2];

    // Initialize the duplicate points array
    uint8_t points_duplicate[TOTAL_NUM_POINTS * 2];

    // Randomly generate the points
    generate_points(points);

    // Copy the points to the duplicate points array
    for (int i = 0; i < TOTAL_NUM_POINTS * 2; i++) {
        points_duplicate[i] = points[i];
    }

    // Print the first 10 points
    for (int i = 0; i < 19; i+=2) {
        printf("Point %d: (%d, %d)\n", i/2, points[i], points[i + 1]);
    }

    // Start the timer
    clock_t start, end;
    start = clock();

    // Generate the centroids' index
    uint16_t centroids[NUM_CENTROIDS];
    for (int i = 0; i < NUM_CENTROIDS; i++){
        // Generate random centroids index
        centroids[i] = rand() % TOTAL_NUM_POINTS;
    }



    /* Initialize the distance matrix: 
            1. Each centroid is at the first elment of each row
            2. The following elements are the distance between the centroid and the points
            3. The distance is not euclidean distance, but the square of the euclidean distance
    */
    uint64_t distance_matrix[DISTANCE_MATRIX_SIZE];

    // Calculate how many points each DPU will handle
    int num_points_per_dpu = TOTAL_NUM_POINTS / DPU_NUMBER;

    // Create a DPU set
    struct dpu_set_t set, dpu;

    // Calculate the distance matrix for each centroid, each time use all dpus to calculate
    for (int i = 0; i < NUM_CENTROIDS; i++) {

        DPU_ASSERT(dpu_alloc(DPU_NUMBER, NULL, &set));

        // Load the distance matrix DPU
        DPU_ASSERT(dpu_load(set, DISTANCE_MATRIX, NULL));
        
        // Populate the data to the DPUs
        populate_mram(set, dpu, points, centroids[i]);

        // Execute the DPU program
        DPU_ASSERT(dpu_launch(set, DPU_SYNCHRONOUS));

        // Initial dpu id
        uint32_t each_dpu;

        // Get the result from the DPUs
        DPU_FOREACH(set, dpu, each_dpu){
            // Prepare the data for each DPU
            DPU_ASSERT(dpu_prepare_xfer(dpu, &distance_matrix[i * (TOTAL_NUM_POINTS + 1) + each_dpu * (num_points_per_dpu + 1)]));
        }

        DPU_ASSERT(dpu_push_xfer(set, DPU_XFER_FROM_DPU, "distance", 0,  (num_points_per_dpu + 1) * sizeof(uint64_t), DPU_XFER_DEFAULT));

        
        // Free the DPUs
        DPU_ASSERT(dpu_free(set));

        // Copy duplicate points to the points array
        for (int j = 0; j < TOTAL_NUM_POINTS * 2; j++) {
            points[j] = points_duplicate[j];
        }
    }

    // End the timer
    end = clock();


    // Find the nearest centroid to each point
    uint16_t nearest_centroid[TOTAL_NUM_POINTS];
    find_nearest_centroid(distance_matrix, nearest_centroid, centroids);

    // Calculate the number of points for each centroid
    uint16_t num_points_per_centroid[NUM_CENTROIDS];
    for (int i = 0; i < TOTAL_NUM_POINTS; i++) {
        num_points_per_centroid[nearest_centroid[i]]++;
    }

    // Initialize the sum variables for each DPU
    uint64_t dpu_sum[NUM_CENTROIDS * 2];

    uint64_t total_sum[NUM_CENTROIDS * 2];

    // Loop through all the centroids
    for (int i = 0; i < NUM_CENTROIDS; i++) {

        // For all 4 centroids, calculate the average coordinate
        // Create a DPU set
        DPU_ASSERT(dpu_alloc(DPU_NUMBER, NULL, &set));

        // Load the average coordinate DPU
        DPU_ASSERT(dpu_load(set, AVG_COORDINATE, NULL));

        // Reconstruc the points array
        
        // Select the points that belong to current centroid
        uint8_t points_centroid[TOTAL_NUM_POINTS * 2];

        // Initialize points_centroid
        for (int j = 0; j < TOTAL_NUM_POINTS * 2; j++) {
            points_centroid[j] = 0;
        }

        // Put the points belong to the current centroid to the points_centroid array
        int index = 0;
        for (int j = 0; j < TOTAL_NUM_POINTS; j++) {
            if (nearest_centroid[j] == i) {
                points_centroid[index] = points[j * 2];
                points_centroid[index + 1] = points[j * 2 + 1];
                index += 2;
            }
        }
        

        // Populate the data to the DPUs
        populate_mram_avg(set, dpu, points_centroid);

        // Execute the DPU program
        DPU_ASSERT(dpu_launch(set, DPU_SYNCHRONOUS));

        

        uint32_t each_dpu;
        // Get the result from the DPUs
        DPU_FOREACH(set, dpu, each_dpu){
            // Prepare the data for each DPU
            DPU_ASSERT(dpu_prepare_xfer(dpu, &dpu_sum[each_dpu * 2]));
        }

        DPU_ASSERT(dpu_push_xfer(set, DPU_XFER_FROM_DPU, "total", 0, 2 * sizeof(uint64_t), DPU_XFER_DEFAULT));
        
        // Free the DPUs
        DPU_ASSERT(dpu_free(set));

        // Calculate the total sum
        total_sum[i * 2] = 0;
        total_sum[i * 2 + 1] = 0;
        for (int j = 0; j < DPU_NUMBER; j++) {
            total_sum[i * 2] += dpu_sum[j * 2];
            total_sum[i * 2 + 1] += dpu_sum[j * 2 + 1];
        }

    }


    // Print the total sum
    for (int i = 0; i < NUM_CENTROIDS; i++) {
        printf("Total sum Centroid %d: (%lu, %lu)\n", i, total_sum[i * 2], total_sum[i * 2 + 1]);
    }
    
    // Calculate the average coordinate for each centroid use total_sum
    int avg[NUM_CENTROIDS * 2];
    for (int i = 0; i < NUM_CENTROIDS; i++) {
        avg[i * 2] = total_sum[i * 2] / num_points_per_centroid[i];
        avg[i * 2 + 1] = total_sum[i * 2 + 1] / num_points_per_centroid[i];
    }

    // Print the average coordinates
    for (int i = 0; i < NUM_CENTROIDS; i++) {
        printf("AVG Centroid %d: (%d, %d)\n", i, avg[i * 2], avg[i * 2 + 1]);
    }

    /*
        Update the centroids index by finding the closest point to the average coordinate in each cluster

    */
    for (int i = 0; i < NUM_CENTROIDS; i++)
    {
        // Loop through all the points belong to the current centroid
        uint64_t min_distance = UINT64_MAX;
        uint16_t min_index = 0;
        for (int j = 0; j < TOTAL_NUM_POINTS; j++)
        {
            if (nearest_centroid[j] == i)
            {
                uint64_t dx = avg[i * 2] - points[j * 2];
                uint64_t dy = avg[i * 2 + 1] - points[j * 2 + 1];
                uint64_t dist = dx * dx + dy * dy;
                if (dist < min_distance)
                {
                    min_distance = dist;
                    min_index = j;
                }
            }
            // Update the centroids
            centroids[i] = min_index;
        }
    }


    // Go in the interactive mode
    int iterations = 9;
    for (int iter = 0; iter < iterations; iter++) {
        // Calculate the distance matrix use DPUs
        for (int i = 0; i < NUM_CENTROIDS; i++) {

            DPU_ASSERT(dpu_alloc(DPU_NUMBER, NULL, &set));

            // Load the distance matrix DPU
            DPU_ASSERT(dpu_load(set, DISTANCE_MATRIX, NULL));
            
            // Populate the data to the DPUs
            populate_mram(set, dpu, points, centroids[i]);

            // Execute the DPU program
            DPU_ASSERT(dpu_launch(set, DPU_SYNCHRONOUS));

            // Initial dpu id
            uint32_t each_dpu;

            // Get the result from the DPUs
            DPU_FOREACH(set, dpu, each_dpu){
                // Prepare the data for each DPU
                DPU_ASSERT(dpu_prepare_xfer(dpu, &distance_matrix[i * (TOTAL_NUM_POINTS + 1) + each_dpu * (num_points_per_dpu + 1)]));
            }

            DPU_ASSERT(dpu_push_xfer(set, DPU_XFER_FROM_DPU, "distance", 0,  (num_points_per_dpu + 1) * sizeof(uint64_t), DPU_XFER_DEFAULT));

            
            // Free the DPUs
            DPU_ASSERT(dpu_free(set));

            // Copy duplicate points to the points array
            for (int j = 0; j < TOTAL_NUM_POINTS * 2; j++) {
                points[j] = points_duplicate[j];
            }
        }

        // Find the nearest centroid to each point
        find_nearest_centroid(distance_matrix, nearest_centroid, centroids);

        // Calculate the number of points for each centroid
        for (int i = 0; i < NUM_CENTROIDS; i++) {
            num_points_per_centroid[i] = 0;
        }

        for (int i = 0; i < TOTAL_NUM_POINTS; i++) {
            num_points_per_centroid[nearest_centroid[i]]++;
        }

        // Initialize the sum variables for each DPU
        for (int i = 0; i < NUM_CENTROIDS; i++) {
            dpu_sum[i * 2] = 0;
            dpu_sum[i * 2 + 1] = 0;
        }

        // Loop through all the centroids
        for (int i = 0; i < NUM_CENTROIDS; i++) {

            // For all 4 centroids, calculate the average coordinate
            // Create a DPU set
            DPU_ASSERT(dpu_alloc(DPU_NUMBER, NULL, &set));

            // Load the average coordinate DPU
            DPU_ASSERT(dpu_load(set, AVG_COORDINATE, NULL));

            // Reconstruc the points array
            
            // Select the points that belong to current centroid
            uint8_t points_centroid[TOTAL_NUM_POINTS * 2];

            // Initialize points_centroid
            for (int j = 0; j < TOTAL_NUM_POINTS * 2; j++) {
                points_centroid[j] = 0;
            }

            // Put the points belong to the current centroid to the points_centroid array
            int index = 0;
            for (int j = 0; j < TOTAL_NUM_POINTS; j++) {
                if (nearest_centroid[j] == i) {
                    points_centroid[index] = points[j * 2];
                    points_centroid[index + 1] = points[j * 2 + 1];
                    index += 2;
                }
            }
            

            // Populate the data to the DPUs
            populate_mram_avg(set, dpu, points_centroid);

            // Execute the DPU program
            DPU_ASSERT(dpu_launch(set, DPU_SYNCHRONOUS));

            

            uint32_t each_dpu;
            // Get the result from the DPUs
            DPU_FOREACH(set, dpu, each_dpu){
                // Prepare the data for each DPU
                DPU_ASSERT(dpu_prepare_xfer(dpu, &dpu_sum[each_dpu * 2]));
            }

            DPU_ASSERT(dpu_push_xfer(set, DPU_XFER_FROM_DPU, "total", 0, 2 * sizeof(uint64_t), DPU_XFER_DEFAULT));
            
            // Free the DPUs
            DPU_ASSERT(dpu_free(set));

            // Calculate the total sum
            total_sum[i * 2] = 0;
            total_sum[i * 2 + 1] = 0;
            for (int j = 0; j < DPU_NUMBER; j++) {
                total_sum[i * 2] += dpu_sum[j * 2];
                total_sum[i * 2 + 1] += dpu_sum[j * 2 + 1];
            }

        }

        // Calculate the average coordinate for each centroid use total_sum
        for (int i = 0; i < NUM_CENTROIDS; i++) {
            avg[i * 2] = total_sum[i * 2] / num_points_per_centroid[i];
            avg[i * 2 + 1] = total_sum[i * 2 + 1] / num_points_per_centroid[i];
        }

        // Print the average coordinates
        for (int i = 0; i < NUM_CENTROIDS; i++) {
            printf("AVG Centroid %d: (%d, %d)\n", i, avg[i * 2], avg[i * 2 + 1]);
        }

        /*
            Update the centroids index by finding the closest point to the average coordinate in each cluster

        */
        for (int i = 0; i < NUM_CENTROIDS; i++)
        {
            // Loop through all the points belong to the current centroid
            uint64_t min_distance = UINT64_MAX;
            uint16_t min_index = 0;
            for (int j = 0; j < TOTAL_NUM_POINTS; j++)
            {
                if (nearest_centroid[j] == i)
                {
                    uint64_t dx = avg[i * 2] - points[j * 2];
                    uint64_t dy = avg[i * 2 + 1] - points[j * 2 + 1];
                    uint64_t dist = dx * dx + dy * dy;
                    if (dist < min_distance)
                    {
                        min_distance = dist;
                        min_index = j;
                    }
                }
                // Update the centroids
                centroids[i] = min_index;
            }
        }

    }

    return 0;
}
