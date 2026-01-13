# Neural Network from Scratch

A clean, pure NumPy implementation of a Feedforward Neural Network (Multi-Layer Perceptron). This project bypasses modern deep learning frameworks (like PyTorch or TensorFlow) to demonstrate the fundamental mathematics behind neural networks, including a manual implementation of the Chain Rule for backpropagation and Stochastic Gradient Descent (SGD).

**Note:** This implementation is currently designed for **binary classification tasks** using the Sigmoid activation function.

## 🚀 Features

* **No Frameworks:** Built entirely using Python and NumPy. No PyTorch, TensorFlow, or Keras.
* **Dynamic Architecture:** Support for any number of layers with arbitrary neuron counts.
* **Backpropagation:** Full manual implementation of the chain rule for gradient calculation.
* **Matrix Operations:** Optimized forward and backward passes using vectorization.
* **Activation:** Sigmoid activation function.

## 🛠️ Prerequisites

* Python 3.x
* NumPy
* Pandas (for data loading)

```bash
pip install numpy pandas
```

## 📝 Usage
1. Initialize the model object.
```Python
model = Network()
```

2. Define architecture. Ensure number of neurons of first layer is equal to the input size. 
```Python
model.add(input_length)
model.add(3)
model.add(1)
```

3. Train the model using the .train() method. You must specify the input data (X), labels (y), and the learning_rate.

```Python 
for _ in range(epochs):
    model.train(X_train, train_labels, 0.1)
```
The training loop handles forward propagation, loss calculation, backpropagation, and weight updates.

4. Optionally, test the model using the .test() method. Pass input data and labels.

```Python
model.test(X_test, test_labels)
```

## 📂 Project Structure
* **Layer Class** 
  * Stores weights, biases, activations, and z values.
  * Handles calc_gradients() for its specific neurons.
  * Updates parameters using weight_gradients and bias_gradients.

* **Network Class**
  * Orchestrates the layers.
  * Manages the Forward Pass loop (Input $\to$ Output).
  * Manages the Backward Pass loop (Output $\to$ Input).

## 🤝 Contributing
Contributions are welcome! If you want to extend this project, consider implementing:

* New Activations: ReLU, Leaky ReLU, or Tanh.

* New Loss Functions: Cross-Entropy Loss.

* Mini-Batch Gradient Descent: Updating weights after a batch of samples instead of every sample.