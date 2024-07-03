import numpy as np
import pandas as pd

# Steering Alignment Neural Network (tanh)
# By Michael Dallalah 6/10/2024
# Last Updated 6/27/2024

# Define the activation function (tanh)
def tanh(x):
    return np.tanh(x)

# Define the derivative of the activation function
def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2

# Load the data from the CSV file
data = pd.read_csv('Data3.csv')

# Split the data into inputs and outputs
X = data.iloc[:, :15].values
y = data.iloc[:, 15:].values

# Initialize the weights randomly
weights1 = np.random.randn(15, 256) #default to (15, 64)
weights2 = np.random.randn(256, 4) #default to (64, 4)

# Define the forward pass function
def forward_pass(X, weights1, weights2):
    # Compute the first layer
    layer1 = tanh(np.dot(X, weights1))
    
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
        layer1 = tanh(np.dot(X, weights1))
        layer2 = np.dot(layer1, weights2)
        
        # Compute the loss
        loss = loss_function(y, layer2)
        
        # Backpropagation
        dLayer2 = layer2 - y
        dLayer1 = np.dot(dLayer2, weights2.T) * tanh_derivative(layer1)
        
        # Update the weights
        weights2 -= learning_rate * np.dot(layer1.T, dLayer2) / m
        weights1 -= learning_rate * np.dot(X.T, dLayer1) / m

        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss:.4f}") #uncomment for larger num_epochs and lower learning_rate

    return weights1, weights2

# Train the model
num_epochs = 100000 #default to 10000
learning_rate = 0.001 #default to 0.001
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