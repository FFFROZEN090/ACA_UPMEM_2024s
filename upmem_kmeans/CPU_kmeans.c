#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define MAX_ITER 100 // Maximum number of iterations

// Calculate the Euclidean distance between two points
double euclidean_distance(double *point1, double *point2, int dimensions)
{
    double distance = 0.0;
    for (int i = 0; i < dimensions; i++)
    {
        distance += pow(point1[i] - point2[i], 2);
    }
    return sqrt(distance);
}

// K-means Algorithsm
void kmeans(double **data, int n, int k, int dimensions, int *labels, double **centroids)
{
    int changed, iterations = 0;
    double **new_centroids = (double **)malloc(k * sizeof(double *));
    int *cluster_sizes = (int *)malloc(k * sizeof(int));

    for (int i = 0; i < k; i++)
    {
        new_centroids[i] = (double *)malloc(dimensions * sizeof(double));
    }

    // Initialize the centroid and select the first k points as the initial centroid
    for (int i = 0; i < k; i++)
    {
        for (int j = 0; j < dimensions; j++)
        {
            centroids[i][j] = data[i][j];
        }
    }

    do
    {
        // Reset temporary variables
        for (int i = 0; i < k; i++)
        {
            cluster_sizes[i] = 0;
            for (int j = 0; j < dimensions; j++)
            {
                new_centroids[i][j] = 0.0;
            }
        }

        // Assign each point to the nearest centroid
        changed = 0;
        for (int i = 0; i < n; i++)
        {
            int closest_centroid = 0;
            double min_distance = euclidean_distance(data[i], centroids[0], dimensions);

            for (int j = 1; j < k; j++)
            {
                double distance = euclidean_distance(data[i], centroids[j], dimensions);
                if (distance < min_distance)
                {
                    min_distance = distance;
                    closest_centroid = j;
                }
            }

            if (labels[i] != closest_centroid)
            {
                changed = 1;
            }
            labels[i] = closest_centroid;
            cluster_sizes[closest_centroid]++;

            for (int j = 0; j < dimensions; j++)
            {
                new_centroids[closest_centroid][j] += data[i][j];
            }
        }

        // Calculate the new centroid
        for (int i = 0; i < k; i++)
        {
            for (int j = 0; j < dimensions; j++)
            {
                centroids[i][j] = new_centroids[i][j] / cluster_sizes[i];
            }
        }

        iterations++;
    } while (changed && iterations < MAX_ITER);

    // Freeing up memory
    for (int i = 0; i < k; i++)
    {
        free(new_centroids[i]);
    }
    free(new_centroids);
    free(cluster_sizes);
}

int main()
{
    int n = 8;
    int dimensions = 2;
    int k = 2; // Number of clusters

    // Sample Data
    double data[8][2] = {
        {1.0, 2.0}, {1.5, 1.8}, {5.0, 8.0}, {8.0, 8.0}, {1.0, 0.6}, {9.0, 11.0}, {8.0, 2.0}, {10.0, 2.0}};

    // Dynamically allocating memory
    double **data_points = (double **)malloc(n * sizeof(double *));
    for (int i = 0; i < n; i++)
    {
        data_points[i] = data[i];
    }

    int *labels = (int *)malloc(n * sizeof(int));
    double **centroids = (double **)malloc(k * sizeof(double *));
    for (int i = 0; i < k; i++)
    {
        centroids[i] = (double *)malloc(dimensions * sizeof(double));
    }

    clock_t start_time = clock();

    // Run K-means
    kmeans(data_points, n, k, dimensions, labels, centroids);

    clock_t end_time = clock();

    // Calculate and output the running time
    double cpu_time_used = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;
    printf("CPU time used: %f seconds\n", cpu_time_used);

    printf("Cluster assignments:\n");
    for (int i = 0; i < n; i++)
    {
        printf("Point %d: Cluster %d\n", i, labels[i]);
    }

    printf("\nFinal centroids:\n");
    for (int i = 0; i < k; i++)
    {
        printf("Centroid %d: (", i);
        for (int j = 0; j < dimensions; j++)
        {
            printf("%f", centroids[i][j]);
            if (j < dimensions - 1)
                printf(", ");
        }
        printf(")\n");
    }
    // Output results to CSV file
    export_results(data_points, labels, n, dimensions);

    // Freeing up memory
    free(labels);
    for (int i = 0; i < k; i++)
    {
        free(centroids[i]);
    }
    free(centroids);
    free(data_points);

    return 0;
}