# Neural Networks: Understanding a Simple Feedforward Neural Network

## Project Overview

In this project, I implemented and analyzed a simple Feedforward Neural Network (FNN) to better understand the inner workings of neural networks, particularly how they learn features, create decision boundaries, and optimize gradients. The focus was on developing a neural network from scratch, using a 2D dataset, and analyzing its behavior throughout the training process.

## Key Features

### 1. **Feedforward Neural Network Architecture**
The neural network was built from scratch with the following architecture:
- **Input Layer:** 2 input neurons corresponding to a 2D dataset.
- **Hidden Layer:** A single hidden layer with 3 neurons.
- **Output Layer:** A single neuron for binary classification (output value 0 or 1).

### 2. **Activation Functions**
The hidden layer used one of three non-linear activation functions: `tanh`, `relu`, or `sigmoid`. These activation functions enable the network to learn complex patterns and help with the backpropagation process:
- **Tanh:** Hyperbolic tangent activation function.
- **ReLU:** Rectified Linear Unit, used for sparsity and better gradient propagation.
- **Sigmoid:** S-shaped curve, often used for binary classification tasks.

### 3. **Loss Function and Optimizer**
- **Loss Function:** I implemented the **cross-entropy loss** function, which is commonly used for binary classification problems.
- **Optimizer:** The gradient descent algorithm was used to minimize the loss function by adjusting the model parameters (weights and biases) during backpropagation.

### 4. **Dataset**
A **randomly generated 2D dataset** was created with two classes, separated by a **circular decision boundary**. This allows for a clear visualization of the model’s decision boundary as it learns.

### 5. **Training Process Visualization**
The project emphasizes understanding the training process by visualizing the learned features, the decision boundary, and the gradients during training. The following visualizations were implemented:
- **Learned Features in Hidden Space:** Visualized how the neural network learns to represent the input space in the hidden layer.
- **Decision Boundary:** Plotted the decision boundary in the input space, showing how the neural network separates the two classes.
- **Gradient Visualization:** Visualized the gradients, with edge thickness indicating the magnitude of the gradient, showing how the network adjusts during training.
- **Animation:** An animation illustrating the entire training process, showing how the decision boundary evolves over time as the network learns.

### 6. **Interactive Web Application**

After implementing the neural network and training visualizations, the project was integrated into an **interactive Flask web application**. This allows users to experiment with various parameters and visually observe the network’s training process.

#### **User Input:**
- The user specifies the desired **activation function** (tanh, ReLU, or sigmoid).
- The user also selects the number of **epochs** and **learning rate** for the training process.
  
#### **Interactive Module:**
- Once the user inputs the parameters, they can click **"Train and Visualize"** to start the training process.
- The resulting figures, including the decision boundary and gradient visualizations, are displayed dynamically.

To run the Flask application locally, the user follows these steps:
```bash
make install   # Installs necessary dependencies
make run       # Starts the Flask application
