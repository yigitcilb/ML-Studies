# iris dataset on K-means cluster algorithm

This project implements a **K-means cluster** algorithm on **iris dataset** in c.

## Files

- `K_means.c` → c implementation of the model  
- `iris.txt` → iris dataset  
---

## Update equation

$$
\mu_i = \frac{1}{|C_i|} \sum_{x \in C_i} x
$$

## Cluster Assignment Rule

$$
C_i = \{ x \mid \| x - \mu_i \|^2 \le \| x - \mu_j \|^2, \ \forall j = 1, \dots, K \}
$$
