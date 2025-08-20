# MNIST Batch Gradient Descent

This project implements a **simple linear regression model** on the MNIST dataset using **batch gradient descent** in C++.  

## Files

- `BatchGradDescentForMINST.cpp` → C++ implementation of the model  
- `train-images-idx3-ubyte` → MNIST training images  
- `train-labels-idx1-ubyte` → MNIST training labels  
- `t10k-images-idx3-ubyte` → MNIST test images  
- `t10k-labels-idx1-ubyte` → MNIST test labels  

---

## Model

Prediction formula:

$$
\hat{y}_i = \mathbf{w}^T \mathbf{x}_i + b
$$

- $\mathbf{x}_i$ → feature vector of the i-th sample (784 pixels)  
- $\mathbf{w}$ → weight vector  
- $b$ → bias  

---

## Error and Gradients

- Error:

$$
\text{error}_i = \hat{y}_i - y_i
$$

- Batch gradients:

$$
\text{grad}_w = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}_i - y_i) \mathbf{x}_i
$$

$$
\text{grad}_b = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}_i - y_i)
$$

- $m$ → number of samples in the batch  

---

## Parameter Update

Weights and bias are updated as:

$$
\mathbf{w} \gets \mathbf{w} - \alpha \, \text{grad}_w
$$

$$
b \gets b - \alpha \, \text{grad}_b
$$

- $\alpha$ → learning rate  

---
