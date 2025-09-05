#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define ROWS 150
#define COLS 4
#define CLASSCOUNT 3

#define DIM 4
#define N 150
#define K 3
#define REP 500
#define MAX_SHOW 10


void kmeans_init(double features[N][DIM], double centroids[K][DIM]) {
    int labels[N];

    for (int k = 0; k < K; k++) {
        int idx = rand() % N;
        for (int d = 0; d < DIM; d++)
            centroids[k][d] = features[idx][d];
    }
}
double DistCalc(double array1[DIM], double array2[DIM]){
    double dist = 0;
    for (int i = 0;i < DIM; i++){
        dist = dist + pow((array1[i] - array2[i]), 2);
    }
    dist = sqrt(dist);
    return dist;
}

void CentroidFitter(double centroids[K][DIM], double features[N][DIM], int labels[N]) {
    

    for (int t = 0; t < REP; t++) {
        for (int i = 0; i < N; i++) {
            double min_distance = DistCalc(centroids[0], features[i]);
            int best_j = 0;

            for (int j = 1; j < K; j++) {
                double d = DistCalc(centroids[j], features[i]);
                if (d < min_distance) {
                    min_distance = d;
                    best_j = j;
                }
            }
            labels[i] = best_j;
        }
        for (int i = 0; i < K; i++) {
            double total[DIM] = {0};
            int count = 0;

            for (int j = 0; j < N; j++) {
                if (labels[j] == i) {
                    for (int d = 0; d < DIM; d++)
                        total[d] += features[j][d];
                    count++;
                }
            }

            if (count > 0) {
                for (int d = 0; d < DIM; d++)
                    centroids[i][d] = total[d] / count;
            }
        }
    }
}
void ShowClusterSamples(double features[N][DIM], int labels[N]) {
    for (int k = 0; k < K; k++) {
        printf("Cluster %d:\n", k);
        int shown = 0;

        for (int i = 0; i < N && shown < MAX_SHOW; i++) {
            if (labels[i] == k) {
                printf("  [");
                for (int d = 0; d < DIM; d++) {
                    printf("%.2f", features[i][d]);
                    if (d < DIM-1) printf(", ");
                }
                printf("]\n");
                shown++;
            }
        }

        if (shown == 0) printf("  (no points in this cluster)\n");
        printf("\n");
    }
}


int main() {
    FILE *fp = fopen("iris.txt", "r");
    if (!fp) { printf("Dosya açılamadı!\n"); return 1; }

    double features[ROWS][COLS];
    int labels[ROWS];
    char line[100];
    int row = 0;

    while (fgets(line, sizeof(line), fp) && row < ROWS) {
        char *token = strtok(line, ",");
        for (int col = 0; col < COLS; col++) {
            features[row][col] = atof(token);
            token = strtok(NULL, ",");
        }

        if (strcmp(token, "Iris-setosa\n") == 0 || strcmp(token, "Iris-setosa") == 0)
            labels[row] = 0;
        else if (strcmp(token, "Iris-versicolor\n") == 0 || strcmp(token, "Iris-versicolor") == 0)
            labels[row] = 1;
        else
            labels[row] = 2;

        row++;
        }
    fclose(fp);

    //initialize K_means clusters
    double centroids[K][DIM];
    kmeans_init(features, centroids);
    int labelss[N];
    CentroidFitter(centroids, features, labelss);
    ShowClusterSamples(features, labelss);

}
