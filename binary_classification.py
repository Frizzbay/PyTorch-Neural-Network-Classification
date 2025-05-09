"""
Binary classification with PyTorch neural networks
"""

import torch
from torch import nn
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import requests
from pathlib import Path

# Check for helper functions
if Path("helper_functions.py").is_file():
    print("helper_function.py already exists")
else:
    print("Downloading helper_functions.py")
    request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/refs/heads/main/helper_functions.py")
    with open("helper_functions.py", "wb") as file:
        file.write(request.content)

from helper_functions import plot_predictions, plot_decision_boundary

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Create binary classification accuracy function
def accuracy_fn(y_true, y_pred):
    """Calculate accuracy between truth labels and predictions."""
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct/len(y_pred)) * 100
    return acc

# 1. Create classification data
def create_circle_data(n_samples=1000, noise=0.03, random_state=42):
    """Create a dataset of circle data using scikit-learn's make_circles."""
    # Make 1000 samples
    X, y = make_circles(n_samples,
                        noise=noise,
                        random_state=random_state)
    
    # Turn data into tensors
    X = torch.from_numpy(X).type(torch.float)
    y = torch.from_numpy(y).type(torch.float)
    
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.2, 
                                                        random_state=random_state)
    
    # Print shapes
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")
    
    # Visualize the data
    plt.figure(figsize=(10, 7))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu)
    plt.title("Circle Classification Dataset")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.savefig("circle_data.png")
    plt.close()
    
    return X_train, X_test, y_train, y_test

# 2. Build models
class CircleModelV0(nn.Module):
    """Simple neural network without non-linearity."""
    def __init__(self):
        super().__init__()
        # Create 2 nn.Linear layers capable of handling the shapes of our data
        self.layer_1 = nn.Linear(in_features=2, out_features=5) 
        self.layer_2 = nn.Linear(in_features=5, out_features=1) 

    def forward(self, x):
        # x -> layer_1 -> layer_2 -> output
        return self.layer_2(self.layer_1(x))

class CircleModelV1(nn.Module):
    """Deeper neural network without non-linearity."""
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2, out_features=10)
        self.layer_2 = nn.Linear(in_features=10, out_features=10)
        self.layer_3 = nn.Linear(in_features=10, out_features=1)

    def forward(self, x):
        # Pass data through each layer
        return self.layer_3(self.layer_2(self.layer_1(x)))

class CircleModelV2(nn.Module):
    """Neural network with non-linear activation functions."""
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2, out_features=10)
        self.layer_2 = nn.Linear(in_features=10, out_features=10)
        self.layer_3 = nn.Linear(in_features=10, out_features=1)
        self.relu = nn.ReLU() # ReLU is a non-linear activation function

    def forward(self, x):
        # Add ReLU activations between layers
        return self.layer_3(self.relu(self.layer_2(self.relu(self.layer_1(x)))))

# 3. Train a model
def train_model(model, X_train, y_train, X_test, y_test, epochs=1000, lr=0.1):
    """Train a PyTorch model."""
    # Set random seeds
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    
    # Loss function and optimizer
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=lr)
    
    # Put data on the target device
    X_train, y_train = X_train.to(device), y_train.to(device)
    X_test, y_test = X_test.to(device), y_test.to(device)
    
    # Training loop
    results = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }
    
    for epoch in range(epochs):
        ### Training
        model.train()

        # 1. Forward pass
        y_logits = model(X_train).squeeze()
        y_pred = torch.round(torch.sigmoid(y_logits)) 

        # 2. Calculate loss/accuracy
        loss = loss_fn(y_logits, y_train)
        acc = accuracy_fn(y_true=y_train, y_pred=y_pred)

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward (back propagation)
        loss.backward()

        # 5. Optimizer step (gradient descent)
        optimizer.step()

        ### Testing
        model.eval()
        with torch.inference_mode():
            # 1. Forward pass
            test_logits = model(X_test).squeeze()
            test_pred = torch.round(torch.sigmoid(test_logits))

            # 2. Calculate test loss/acc
            test_loss = loss_fn(test_logits, y_test)
            test_acc = accuracy_fn(y_true=y_test, y_pred=test_pred)
        
        # Store results
        results["train_loss"].append(loss.item())
        results["train_acc"].append(acc)
        results["test_loss"].append(test_loss.item())
        results["test_acc"].append(test_acc)

        # Print progress
        if epoch % 100 == 0:
            print(f"Epoch: {epoch} | Loss: {loss:.5f} | Accuracy: {acc:.2f}% | Test Loss: {test_loss:.5f} | Test Accuracy: {test_acc:.2f}%")
    
    return results

# 4. Plot decision boundaries
def plot_model_results(model_name, model, X_train, y_train, X_test, y_test):
    """Plot decision boundaries for a given model."""
    # Plot decision boundaries
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title(f"{model_name} - Train Data")
    plot_decision_boundary(model, X_train, y_train)
    plt.subplot(1, 2, 2)
    plt.title(f"{model_name} - Test Data")
    plot_decision_boundary(model, X_test, y_test)
    plt.savefig(f"{model_name}_boundaries.png")
    plt.close()

# 5. Compare models with and without non-linearity
def compare_models(models_dict, X_train, y_train, X_test, y_test):
    """Compare multiple models on the same data."""
    # Plot all models
    plt.figure(figsize=(15, 10))
    i = 1
    for name, model in models_dict.items():
        plt.subplot(len(models_dict), 2, i)
        plt.title(f"{name} - Train")
        plot_decision_boundary(model, X_train, y_train)
        i += 1
        plt.subplot(len(models_dict), 2, i)
        plt.title(f"{name} - Test")
        plot_decision_boundary(model, X_test, y_test)
        i += 1
    plt.tight_layout()
    plt.savefig("model_comparison.png")
    plt.close()

# 6. Demonstrate ReLU and Sigmoid activation functions
def plot_activation_functions():
    """Plot ReLU and Sigmoid activation functions."""
    # Create a tensor
    A = torch.arange(-10, 11, 1, dtype=torch.float32)
    
    # Define ReLU manually
    def relu(x: torch.Tensor) -> torch.Tensor:
        return torch.maximum(torch.tensor(0), x)
    
    # Define sigmoid manually
    def sigmoid(x):
        return 1 / (1 + torch.exp(-x))
    
    # Plot activation functions
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.title("ReLU Activation Function")
    plt.plot(A.numpy(), relu(A).numpy())
    plt.grid(True)
    plt.xlabel("Input")
    plt.ylabel("Output")
    
    plt.subplot(1, 2, 2)
    plt.title("Sigmoid Activation Function")
    plt.plot(A.numpy(), sigmoid(A).numpy())
    plt.grid(True)
    plt.xlabel("Input")
    plt.ylabel("Output")
    
    plt.savefig("activation_functions.png")
    plt.close()

def main():
    """Run the binary classification experiment."""
    # 1. Create data
    X_train, X_test, y_train, y_test = create_circle_data()
    
    # 2. Create models
    print("\nTraining model without non-linearity (V0)...")
    model_0 = CircleModelV0().to(device)
    results_0 = train_model(model_0, X_train, y_train, X_test, y_test)
    
    print("\nTraining deeper model without non-linearity (V1)...")
    model_1 = CircleModelV1().to(device)
    results_1 = train_model(model_1, X_train, y_train, X_test, y_test)
    
    print("\nTraining model with non-linearity (V2)...")
    model_2 = CircleModelV2().to(device)
    results_2 = train_model(model_2, X_train, y_train, X_test, y_test)
    
    # 3. Plot individual model results
    plot_model_results("Model_V0", model_0, X_train, y_train, X_test, y_test)
    plot_model_results("Model_V1", model_1, X_train, y_train, X_test, y_test)
    plot_model_results("Model_V2", model_2, X_train, y_train, X_test, y_test)
    
    # 4. Compare all models
    models_dict = {
        "V0 (No non-linearity)": model_0,
        "V1 (Deeper, no non-linearity)": model_1,
        "V2 (With non-linearity)": model_2
    }
    compare_models(models_dict, X_train, y_train, X_test, y_test)
    
    # 5. Plot activation functions
    plot_activation_functions()
    
    print("\nExperiment complete. Check the generated images for visualization of results.")

if __name__ == "__main__":
    main()
