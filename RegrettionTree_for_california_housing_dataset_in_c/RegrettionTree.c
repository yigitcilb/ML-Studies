#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ROWS 20640
#define COLS 9
#define TEST_RATIO 0.2
#define MAX_DEPTH 9


typedef struct Node{
    int feature_index;
    float threshold;
    float value;
    struct Node *left;
    struct Node *right;
    int is_leaf;
} Node;

float predict(Node *node, float *x) {
    if (node->is_leaf) return node->value;
    if (x[node->feature_index] <= node->threshold)
        return predict(node->left, x);
    else
        return predict(node->right, x);
}

float mean(float *y, int n) {
    float sum = 0;
    for (int i = 0; i < n; i++) sum += y[i];
    return sum / n;
}

float mse(float *y, int n){
    if (n == 0) return 0;
    float m = mean(y, n);
    float err = 0;
    for (int i = 0; i < n; i++) {
        float d = y[i] - m;
        err += d * d;
    }
    return err / n;
}

Node* build_tree(float X[][COLS], float y[], int n, int d, int max_depth){
    Node* node = malloc(sizeof(Node));

    if ( d >= max_depth || n <= 5){
        node->is_leaf = 1;
        node->value = mean(y,n);
        node->left = NULL;
        node->right = NULL;
        return node;
    }

    int best_feature = -1;
    float best_thresh = 0;
    float best_error = 0;
    for ( int f = 0; f < COLS; f++){
        for (int i = 0;i < n; i++){
            float t = X[i][f];
            float left_y[n], right_y[n];
            int nl = 0, nr = 0;
            for (int j = 0; j < n; j++) {
                if (X[j][f] <= t) left_y[nl++] = y[j];
                else right_y[nr++] = y[j];
            }

            float error = (nl * mse(left_y, nl) + nr * mse(right_y, nr)) / n;
            if (error < best_error) {
                best_error = error;
                best_feature = f;
                best_thresh = t;
            }
        }    
    }

    if (best_feature == -1) {
        node->is_leaf = 1;
        node->value = mean(y, n);
        node->left = node->right = NULL;
        return node;
    }
    node->is_leaf = 0;
    node->feature_index = best_feature;
    node->threshold = best_thresh;

    float left_X[n][COLS], right_X[n][COLS], left_y[n], right_y[n];
    int nl = 0, nr = 0;
    for (int i = 0; i < n; i++) {
        if (X[i][best_feature] <= best_thresh) {
            for (int f = 0; f < 9; f++) left_X[nl][f] = X[i][f];
            left_y[nl++] = y[i];
        } else {
            for (int f = 0; f < 9; f++) right_X[nr][f] = X[i][f];
            right_y[nr++] = y[i];
        }
    }

    node->left = build_tree(left_X, left_y, nl, d + 1, max_depth);
    node->right = build_tree(right_X, right_y, nr, d + 1, max_depth);

    return node;
}

int encode_ocean(const char *str) {
    if (strcmp(str, "<1H OCEAN") == 0) return 0;
    if (strcmp(str, "INLAND") == 0) return 1;
    if (strcmp(str, "NEAR BAY") == 0) return 2;
    if (strcmp(str, "NEAR OCEAN") == 0) return 3;
    if (strcmp(str, "ISLAND") == 0) return 4;
    return -1; 
}

int main() {
    FILE *fp = fopen("housing.csv", "r");
    if (!fp) {
        perror("File opening failed");
        return 1;
    }

    float features[ROWS][COLS];
    float target[ROWS];
    char line[1024];

    fgets(line, sizeof(line), fp);

    int row = 0;
    while (fgets(line, sizeof(line), fp) && row < ROWS) {
        float lon, lat, age, rooms, bedrooms, pop, households, income, value;
        char ocean[32];

        if (sscanf(line, "%f,%f,%f,%f,%f,%f,%f,%f,%f,%31[^\n]",
                   &lon, &lat, &age, &rooms, &bedrooms,
                   &pop, &households, &income, &value, ocean) == 10) {
            
            features[row][0] = lon;
            features[row][1] = lat;
            features[row][2] = age;
            features[row][3] = rooms;
            features[row][4] = bedrooms;
            features[row][5] = pop;
            features[row][6] = households;
            features[row][7] = income;
            features[row][8] = encode_ocean(ocean); 
            
            target[row] = value;
            row++;
        }
    }

    fclose(fp);

    float X[20640][COLS];   
    float y[20640];       
    int n = 20640;


    int train_size = (int)(ROWS * (1 - TEST_RATIO));
    int test_size = ROWS - train_size;

    float (*X_train)[COLS] = features;
    float *y_train = target;

    float (*X_test)[COLS] = features + train_size;
    float *y_test = target + train_size;

    Node* tree = build_tree(X_train, y_train, train_size, 0, MAX_DEPTH);

    float total_error = 0;
    for (int i = 0; i < test_size; i++) {
        float pred = predict(tree, X_test[i]);
        float diff = pred - y_test[i];
        total_error += diff * diff;
    }

    float test_mse = total_error / test_size;
    printf("Test MSE: %f\n", test_mse);

    return 0;
}
