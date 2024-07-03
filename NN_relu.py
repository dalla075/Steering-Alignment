import numpy as np
import pandas as pd

# Steering Alignment Neural Network (ReLu)
# By Michael Dallalah 6/6/2024
# Last Updated 6/10/2024

# Define the activation function (ReLU)
def relu(x):
    return np.maximum(0, x)

# Define the derivative of the activation function
def relu_derivative(x):
    return (x > 0).astype(float)

# Load the data from the CSV file
data = pd.read_csv('Data.csv')

# Split the data into inputs and outputs
X = data.iloc[:, :15].values
y = data.iloc[:, 15:].values

# Initialize the weights randomly
weights1 = np.random.randn(15, 2048) #default to (15, 64)
weights2 = np.random.randn(2048, 4) #default to (64, 4)

# Define the forward pass function
def forward_pass(X, weights1, weights2):
    # Compute the first layer
    layer1 = relu(np.dot(X, weights1))
    
    # Compute the second layer
    layer2 = np.dot(layer1, weights2)
    
    return layer2

# Define the loss function (mean squared error)
def loss_function(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Define the gradient descent function
def gradient_descent(X, y, weights1, weights2, learning_rate, num_epochs):
    m = len(X)
    
    for epoch in range(num_epochs):
        # Forward pass
        layer1 = relu(np.dot(X, weights1))
        layer2 = np.dot(layer1, weights2)
        
        # Compute the loss
        loss = loss_function(y, layer2)
        
        # Backpropagation
        dLayer2 = layer2 - y
        dLayer1 = np.dot(dLayer2, weights2.T) * relu_derivative(layer1)
        
        # Update the weights
        weights2 -= learning_rate * np.dot(layer1.T, dLayer2) / m
        weights1 -= learning_rate * np.dot(X.T, dLayer1) / m

        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss:.4f}")

        return weights1, weights2

# Train the model
num_epochs = 100000000 #Default to 1000
learning_rate = 0.001 #Default to 0.001
weights1, weights2 = gradient_descent(X, y, weights1, weights2, learning_rate, num_epochs)

# Allow the user to input new data and get the predicted outputs
while True:
    user_input = input("Enter 15 input values separated by spaces (or 'q' to quit): ")
    if user_input.lower() == 'q':
        break
    
    input_values = [float(x) for x in user_input.split()]
    if len(input_values) != 15:
        print("Invalid input. Please enter 15 values.")
        continue
    
    new_input = np.array([input_values])
    predicted_output = forward_pass(new_input, weights1, weights2)
    print("Predicted output:", predicted_output[0])