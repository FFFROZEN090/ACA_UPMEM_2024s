#include <defs.h>
#include <mram.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

#define MAX_ITERATIONS 100
#define NUM_CENTROIDS 4
#define NUM_FEATURES 2
#define POINTS_PER_DPU 1024
#define FLT_MAX 3.402823466e+38F

__mram_noinit float points[POINTS_PER_DPU][NUM_FEATURES];
__mram_noinit float centroids[NUM_CENTROIDS][NUM_FEATURES];
__mram_noinit int assignment[POINTS_PER_DPU];
__host float host_centroids[NUM_CENTROIDS][NUM_FEATURES];

void compute_assignment(float point[NUM_FEATURES], float centroids[NUM_CENTROIDS][NUM_FEATURES], int* assignment) {
    float min_dist = FLT_MAX;
    int min_idx = 0;

    for (int i = 0; i < NUM_CENTROIDS; i++) {
        float dist = 0.0;
        for (int j = 0; j < NUM_FEATURES; j++) {
            float diff = point[j] - centroids[i][j];
            dist += diff * diff;
        }
        if (dist < min_dist) {
            min_dist = dist;
            min_idx = i;
        }
    }
    *assignment = min_idx;
}

void update_centroids(float points[POINTS_PER_DPU][NUM_FEATURES], int assignment[POINTS_PER_DPU], float centroids[NUM_CENTROIDS][NUM_FEATURES]) {
    float local_sums[NUM_CENTROIDS][NUM_FEATURES] = {0};
    int local_counts[NUM_CENTROIDS] = {0};

    for (int i = 0; i < POINTS_PER_DPU; i++) {
        int cluster = assignment[i];
        for (int j = 0; j < NUM_FEATURES; j++) {
            local_sums[cluster][j] += points[i][j];
        }
        local_counts[cluster]++;
    }

    for (int i = 0; i < NUM_CENTROIDS; i++) {
        for (int j = 0; j < NUM_FEATURES; j++) {
            if (local_counts[i] > 0) {
                centroids[i][j] = local_sums[i][j] / local_counts[i];
            }
        }
    }
}

int main() {
    int iteration = 0;
    bool converged = false;

    while (iteration < MAX_ITERATIONS && !converged) {
        // Assign each point to the nearest centroid
        for (int i = 0; i < POINTS_PER_DPU; i++) {
            compute_assignment(points[i], centroids, &assignment[i]);
        }

        // Update centroids
        update_centroids(points, assignment, centroids);

        // Check for convergence (this can be done on the host)
        // Transfer centroids to host and check if they have changed significantly
        // If not, set converged = true

        iteration++;
    }

    return 0;
}
