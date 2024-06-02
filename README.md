# Neural Network Model for Logical Operations

This project implements a simple neural network in C to learn logical operations such as AND, OR, NAND, and XOR. It utilizes a single-layer perceptron model with a sigmoid activation function to perform binary classification tasks.

## Features

- Implements the sigmoid activation function for neuron activation.
- Uses a simple perceptron model to learn weights and biases for different logical operations.
- Provides functions for computing cost using mean squared error, gradient calculation using both finite difference and backpropagation methods.
- Includes random initialization of weights and biases.
- Detailed console output for tracking learning progress and final model performance.

## Getting Started

### Prerequisites

Ensure you have a C compiler installed on your system, such as GCC for Linux/Mac or Visual Studio for Windows.

### Compilation

To compile the code, use the following command:

```bash
gcc -o neural_network main.c -lm
```

This command compiles the `main.c` file into an executable named `neural_network`, linking the math library with `-lm` for functions like `expf`.

### Running the Program

After compilation, you can run the program by executing:

```bash
./neural_network
```

This will execute the training process for the neural network, displaying the cost and weights periodically, and finally showing the performance of the trained model on a set of test inputs.

## Code Structure

- `sigmoid`: The sigmoid activation function, which transforms input into an output between 0 and 1, making it suitable for binary classification.
- `cost`: Computes the mean squared error across all training examples to evaluate the performance of the model.
- `compute_gradients_finite_difference`: Calculates the gradients of the cost function with respect to weights and bias using the finite difference method.
- `compute_gradients_backprop`: Computes gradients using the backpropagation technique, leveraging the chain rule for derivatives in a more efficient manner than finite differences.
- `random_float`: Utility function to generate a random floating point number between 0 and 1.

## How It Works

The neural network is trained using gradient descent, where the gradients can be calculated either through finite differences or backpropagation. During each iteration of training:

1. The cost is computed based on current model parameters (weights and bias).
2. Gradients are calculated.
3. Model parameters are updated based on gradients to minimize the cost function.

By iterating this process, the model learns to predict the correct outputs for logical operations like AND, OR, NAND, and XOR based on binary inputs.
