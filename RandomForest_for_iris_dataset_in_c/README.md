# iris dataset on random forest algorithm

This project implements a **Random Forest** algorithm on **iris dataset** in c.

## Files

- `RandomForest.c` → c implementation of the model  
- `iris.txt` → iris dataset  
---

## Entropy

Entropy formula:

$$
Entropy(D) = - \sum_{k=1}^{K} p_k \log_2 p_k
$$
---

## Bootstrap Sampling

$$
D_b = \{ (x_i, y_i) \}_{i=1}^{n}, \quad D_b \sim D
$$
---

## Random Forest Desicion

Majority voting:

$$
H(x) = \operatorname{mode}\{ h_1(x), h_2(x), \dots, h_B(x) \}
$$
---