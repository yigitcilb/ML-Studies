#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define ROWS 150
#define COLS 4
#define CLASSCOUNT 3

void best_split(float features[ROWS][COLS], int labels[ROWS], int n_rows, int n_cols, int *best_feature, float *best_threshold);

typedef struct Node {
    int is_leaf;
    int prediction;       
    int feature;          
    float threshold;      
    struct Node *left;
    struct Node *right;
} Node;

Node* build_tree(float features[ROWS][COLS], int labels[ROWS], int n_rows, int n_cols, int depth, int max_depth) {
    int first_label = labels[0];
    int same = 1;
    for (int i = 1; i < n_rows; i++)
        if (labels[i] != first_label) same = 0;

    Node *node = (Node*)malloc(sizeof(Node));

    if (same || depth == max_depth) {
        node->is_leaf = 1;
        node->prediction = first_label; 
        node->left = node->right = NULL;
        return node;
    }

    int best_feature;
    float best_threshold;
    best_split(features, labels, n_rows, n_cols, &best_feature, &best_threshold);
    node->is_leaf = 0;
    node->feature = best_feature;
    node->threshold = best_threshold;

    int left_indices[ROWS], right_indices[ROWS], l_count = 0, r_count = 0;
    for (int i = 0; i < n_rows; i++) {
        if (features[i][best_feature] <= best_threshold)
            left_indices[l_count++] = i;
        else
            right_indices[r_count++] = i;
    }

    float left_features[ROWS][COLS], right_features[ROWS][COLS];
    int left_labels[ROWS], right_labels[ROWS];
    for (int i = 0; i < l_count; i++) {
        left_labels[i] = labels[left_indices[i]];
        for (int j = 0; j < n_cols; j++)
            left_features[i][j] = features[left_indices[i]][j];
    }
    for (int i = 0; i < r_count; i++) {
        right_labels[i] = labels[right_indices[i]];
        for (int j = 0; j < n_cols; j++)
            right_features[i][j] = features[right_indices[i]][j];
    }

    node->left = build_tree(left_features, left_labels, l_count, n_cols, depth + 1, max_depth);
    node->right = build_tree(right_features, right_labels, r_count, n_cols, depth + 1, max_depth);

    return node;
}

int tree_predict(Node *node, float sample[COLS]) {
    if (node->is_leaf) return node->prediction;
    if (sample[node->feature] <= node->threshold)
        return tree_predict(node->left, sample);
    else
        return tree_predict(node->right, sample);
}


double entropy(int counts[], int classCount, int total) {
    double H = 0.0;
    for (int i = 0; i < classCount; i++) {
        if (counts[i] == 0) continue;
        double p = (double)counts[i] / total;
        H -= p * log2(p);
    }
    return H;
}

double information_gain(double parent_entropy,
                        int left_counts[], int left_total,
                        int right_counts[], int right_total,
                        int classCount) {
    double left_weight = (double)left_total / (left_total + right_total);
    double right_weight = (double)right_total / (left_total + right_total);
    double child_entropy =
        left_weight * entropy(left_counts, classCount, left_total) +
        right_weight * entropy(right_counts, classCount, right_total);
    return parent_entropy - child_entropy;
}

void best_split(float features[ROWS][COLS], int labels[ROWS],
                int n_rows, int n_cols,
                int *best_feature, float *best_threshold) {
    int counts[CLASSCOUNT] = {0};
    for (int i = 0; i < n_rows; i++) counts[labels[i]]++;
    double parent_entropy = entropy(counts, CLASSCOUNT, n_rows);

    double max_IG = -1.0;
    *best_feature = 0;
    *best_threshold = 0.0;

    for (int f = 0; f < n_cols; f++) {
        for (float t = 4.0; t <= 8.0; t += 0.1) {
            int l_counts[CLASSCOUNT] = {0}, r_counts[CLASSCOUNT] = {0};
            int l_total = 0, r_total = 0;

            for (int i = 0; i < n_rows; i++) {
                if (features[i][f] <= t) {
                    l_counts[labels[i]]++;
                    l_total++;
                } else {
                    r_counts[labels[i]]++;
                    r_total++;
                }
            }

            if (l_total == 0 || r_total == 0) continue;

            double IG = information_gain(parent_entropy, l_counts, l_total, r_counts, r_total, CLASSCOUNT);
            if (IG > max_IG) {
                max_IG = IG;
                *best_feature = f;
                *best_threshold = t;
            }
        }
    }
}

int predict(float sample[COLS], int best_feature, float best_threshold, int left_class, int right_class) {
    return (sample[best_feature] <= best_threshold) ? left_class : right_class;
}

int main() {
    FILE *fp = fopen("iris.txt", "r");
    if (!fp) { printf("Dosya açılamadı!\n"); return 1; }

    float features[ROWS][COLS];
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

    int train_size = 120;
    int test_size = ROWS - train_size;

    float train_features[train_size][COLS];
    int train_labels[train_size];
    float test_features[test_size][COLS];
    int test_labels[test_size];

    for (int i = 0; i < train_size; i++) {
        train_labels[i] = labels[i];
        for (int j = 0; j < COLS; j++)
            train_features[i][j] = features[i][j];
    }
    for (int i = 0; i < test_size; i++) {
        test_labels[i] = labels[train_size + i];
        for (int j = 0; j < COLS; j++)
            test_features[i][j] = features[train_size + i][j];
    }

    int n_trees = 5;
    int max_depth = 3;
    Node *trees[n_trees];

    srand(42); 
    for (int t = 0; t < n_trees; t++) {
        float bootstrap_features[train_size][COLS];
        int bootstrap_labels[train_size];

        for (int i = 0; i < train_size; i++) {
            int idx = rand() % train_size;
            bootstrap_labels[i] = train_labels[idx];
            for (int j = 0; j < COLS; j++)
                bootstrap_features[i][j] = train_features[idx][j];
        }

        trees[t] = build_tree(bootstrap_features, bootstrap_labels, train_size, COLS, 0, max_depth);
    }

    int correct = 0;
    for (int i = 0; i < test_size; i++) {
        int votes[CLASSCOUNT] = {0};

        for (int t = 0; t < n_trees; t++) {
            int pred = tree_predict(trees[t], test_features[i]);
            votes[pred]++;
        }

        int final_pred = 0;
        for (int c = 1; c < CLASSCOUNT; c++)
            if (votes[c] > votes[final_pred]) final_pred = c;

        if (final_pred == test_labels[i]) correct++;
        printf("Sample %d predicted: %d, actual: %d\n", i, final_pred, test_labels[i]);
    }

    printf("Test accuracy: %.2f%%\n", 100.0 * correct / test_size);
}
