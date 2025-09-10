# housing dataset on regrettion tree algorithm

This project implements a **Regression Tree** algorithm on **housing dataset** in c.

## Files

- `RegrettionTree.c` → c implementation of the model  
- `housing.csv` → housing dataset  
---

## Leaf Node Prediction

$$
\hat{y}_m = \frac{1}{|R_m|} \sum_{i \in R_m} y_i
$$

## Optimal Split Selection

$$
\text{Split}^* = \arg\min_{j, s} \left[ \sum_{i: x_i \in R_1(j,s)} (y_i - \hat{y}_{R_1})^2 + \sum_{i: x_i \in R_2(j,s)} (y_i - \hat{y}_{R_2})^2 \right]
$$