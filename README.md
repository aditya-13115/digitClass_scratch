# Neural Network for MNIST Classification

## Overview
This project implements a **fully connected neural network** from scratch using **NumPy** to classify handwritten digits from the **MNIST dataset**. The model uses **ReLU activation** for hidden layers and **Softmax activation** for the output layer. The training is performed using **gradient descent**.

## Dataset
We use the **MNIST dataset**, which consists of **60,000 training images** and **10,000 test images**, each of size **28x28 pixels**. The dataset is loaded using **Keras API**.

## Features
- **Fully Connected Neural Network** with **One Hidden Layer**
- **ReLU Activation** in the hidden layer
- **Softmax Activation** in the output layer
- **One-Hot Encoding** for labels
- **Gradient Descent Optimization**
- **Accuracy Calculation**
- **Shuffling of Training Data**
- **Visualization of Predictions**

## Installation
Ensure you have Python installed along with the required libraries:
```sh
pip install numpy matplotlib tensorflow
```

## Code Structure
### 1. **Loading MNIST Dataset**
```python
from tensorflow.keras.datasets import mnist
import numpy as np

(x_train, y_train), (x_test, y_test) = mnist.load_data()
```
- Images are reshaped and normalized to **[0,1]**.
- Labels are converted to one-hot encoded vectors.

### 2. **Preprocessing Data**
```python
x_train = x_train.reshape(60000, 28 * 28).T / 255.0
x_test = x_test.reshape(10000, 28 * 28).T / 255.0
y_train = y_train.flatten()
y_test = y_test.flatten()
```

### 3. **Initializing Parameters**
```python
def init_param():
    w1 = np.random.randn(10, 784) * np.sqrt(2 / 784)
    b1 = np.zeros((10, 1))
    w2 = np.random.randn(10, 10) * np.sqrt(2 / 10)
    b2 = np.zeros((10, 1))
    return w1, b1, w2, b2
```

### 4. **Forward Propagation**
```python
def forward_prop(w1, b1, w2, b2, X):
    z1 = w1.dot(X) + b1
    a1 = np.maximum(0, z1)
    z2 = w2.dot(a1) + b2
    a2 = np.exp(z2) / np.sum(np.exp(z2), axis=0)
    return z1, a1, z2, a2
```

### 5. **Backward Propagation & Parameter Update**
```python
def back_prop(z1, a1, z2, a2, w2, X, Y):
    m = Y.size
    one_hot_Y = np.zeros((10, m))
    one_hot_Y[Y, np.arange(m)] = 1
    dz2 = a2 - one_hot_Y
    dw2 = 1/m * dz2.dot(a1.T)
    db2 = np.sum(dz2, axis=1, keepdims=True) / m
    dz1 = w2.T.dot(dz2) * (z1 > 0)
    dw1 = 1/m * dz1.dot(X.T)
    db1 = np.sum(dz1, axis=1, keepdims=True) / m
    return dw1, db1, dw2, db2
```

### 6. **Training the Model**
```python
def gradient_descent(X, Y, iter, alpha):
    w1, b1, w2, b2 = init_param()
    for i in range(iter):
        z1, a1, z2, a2 = forward_prop(w1, b1, w2, b2, X)
        dw1, db1, dw2, db2 = back_prop(z1, a1, z2, a2, w2, X, Y)
        w1 -= alpha * dw1
        b1 -= alpha * db1
        w2 -= alpha * dw2
        b2 -= alpha * db2
        if i % 50 == 0:
            print(f"Iteration {i}, Accuracy: {np.mean(np.argmax(a2, axis=0) == Y)}")
    return w1, b1, w2, b2
```

### 7. **Making Predictions**
```python
def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    return np.argmax(A2, axis=0)
```

### 8. **Testing Predictions with Visualization**
```python
import matplotlib.pyplot as plt

def test_prediction(index, W1, b1, W2, b2):
    current_image = x_train[:, index, None]
    prediction = make_predictions(current_image, W1, b1, W2, b2)
    label = y_train[index]
    print(f"Prediction: {prediction}, Label: {label}")
    plt.imshow(current_image.reshape(28, 28) * 255, cmap='gray')
    plt.show()
```

### 9. **Running the Model**
```python
W1, b1, W2, b2 = gradient_descent(x_train, y_train, 500, 0.1)
```

### 10. **Testing on Some Images**
```python
test_prediction(0, W1, b1, W2, b2)
test_prediction(1, W1, b1, W2, b2)
test_prediction(2, W1, b1, W2, b2)
test_prediction(3, W1, b1, W2, b2)
```

## Results
- The model improves accuracy over training iterations.
- It makes predictions on test images and visualizes them.
