#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <time.h>

#define MAX_ITER 100

// Calcute the distance between 2 points
double euclidean_distance(double* point1, double* point2, int dimensions) {
    double distance = 0.0;
    for (int i = 0; i < dimensions; i++) {
        distance += pow(point1[i] - point2[i], 2);
    }
    return sqrt(distance);
}

// centroid initialization
void initialize_centroids(double** data, double** centroids, int n_points, int k, int dimensions) {
    srand(time(NULL)); //create random number array 
    for (int i = 0; i < k; i++) {
        int random_index = rand() % n_points;
        for (int j = 0; j < dimensions; j++) {
            centroids[i][j] = data[random_index][j];
        }
    }
}

// assign every data point to the closest centroid
void assign_clusters(double** data, double** centroids, int* labels, int n_points, int k, int dimensions) {
    for (int i = 0; i < n_points; i++) {
        double min_distance = DBL_MAX;
        int closest_centroid = 0;
        for (int j = 0; j < k; j++) {
            double distance = euclidean_distance(data[i], centroids[j], dimensions);
            if (distance < min_distance) {
                min_distance = distance;
                closest_centroid = j;
            }
        }
        labels[i] = closest_centroid;
    }
}

// update centroids
void update_centroids(double** data, double** centroids, int* labels, int n_points, int k, int dimensions) {
    int* count = (int*)calloc(k, sizeof(int));
    double** new_centroids = (double**)calloc(k, sizeof(double*));

    for (int i = 0; i < k; i++) {
        new_centroids[i] = (double*)calloc(dimensions, sizeof(double));
    }

    for (int i = 0; i < n_points; i++) {
        int cluster_id = labels[i];
        count[cluster_id]++;
        for (int j = 0; j < dimensions; j++) {
            new_centroids[cluster_id][j] += data[i][j];
        }
    }

    for (int i = 0; i < k; i++) {
        for (int j = 0; j < dimensions; j++) {
            if (count[i] != 0) {
                centroids[i][j] = new_centroids[i][j] / count[i];
            }
        }
    }

    // release memory
    for (int i = 0; i < k; i++) {
        free(new_centroids[i]);
    }
    free(new_centroids);
    free(count);
}

// check the convergence of the centroids
int check_convergence(double** centroids, double** previous_centroids, int k, int dimensions, double tolerance) {
    for (int i = 0; i < k; i++) {
        if (euclidean_distance(centroids[i], previous_centroids[i], dimensions) > tolerance) {
            return 0; // keep iterating
        }
    }
    return 1; // convergent
}

// K-means main function
void kmeans(double** data, int n_points, int dimensions, int k) {
    double** centroids = (double**)malloc(k * sizeof(double*));
    double** previous_centroids = (double**)malloc(k * sizeof(double*));
    int* labels = (int*)malloc(n_points * sizeof(int));
    for (int i = 0; i < k; i++) {
        centroids[i] = (double*)malloc(dimensions * sizeof(double));
        previous_centroids[i] = (double*)malloc(dimensions * sizeof(double));
    }

    initialize_centroids(data, centroids, n_points, k, dimensions);

    clock_t start_time = clock() //start time

    for (int iter = 0; iter < MAX_ITER; iter++) {
        assign_clusters(data, centroids, labels, n_points, k, dimensions);

        // store the present centroids
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < dimensions; j++) {
                previous_centroids[i][j] = centroids[i][j];
            }
        }

        update_centroids(data, centroids, labels, n_points, k, dimensions);

        if (check_convergence(centroids, previous_centroids, k, dimensions, 1e-4)) {
            printf("Convergence after %d iterations\n", iter + 1);
            break;
        }
    }

    clock_t end_time = clock() //end time

    // calculate convergence time
    double convergence_time = (double)(end_time - start_time) / CLOCK_PER_SEC;
    printf("Converging time of K-Means clustering on CPU: %d\n", convergence_time)

    // print the result of clustering
    for (int i = 0; i < n_points; i++) {
        printf("Data Point %d is in cluster %d\n", i, labels[i]);
    }

    // release memory
    for (int i = 0; i < k; i++) {
        free(centroids[i]);
        free(previous_centroids[i]);
    }
    free(centroids);
    free(previous_centroids);
    free(labels);
}

int main() {
    int n_points = 8;         // num. of data points
    int dimensions = 2;       // data dimension
    int k = 2;                // num. of clustering
    double tolerance = 1e-4;  //tolerence of the convergence

    // initialization of the data
    double data_array[8][2] = {
        {1.0, 2.0}, {1.5, 1.8}, {5.0, 8.0}, {8.0, 8.0},
        {1.0, 0.6}, {9.0, 11.0}, {8.0, 2.0}, {10.0, 2.0}
    };

    double** data = (double**)malloc(n_points * sizeof(double*));
    for (int i = 0; i < n_points; i++) {
        data[i] = data_array[i];
    }

    kmeans(data, n_points, dimensions, k);

    free(data);
    return 0;
}