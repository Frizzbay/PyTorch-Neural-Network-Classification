"""
Regression validation for neural network architectures
Used to test if the neural network architecture can at least fit a straight line.
"""

import torch
from torch import nn
import matplotlib.pyplot as plt
from helper_functions import plot_predictions

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 1. Create simple regression data
def create_regression_data(weight=0.7, bias=0.3, start=0, end=1, step=0.01):
    """Create a simple linear regression dataset."""
    # Create data
    X = torch.arange(start, end, step).unsqueeze(dim=1)
    y = weight * X + bias  # Linear regression formula (y = w*X + b)
    
    # Create train and test splits
    train_split = int(0.8 * len(X))
    X_train, y_train = X[:train_split], y[:train_split]
    X_test, y_test = X[train_split:], y[train_split:]
    
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")
    
    # Visualize data
    plt.figure(figsize=(10, 7))
    plt.scatter(X_train, y_train, c="b", s=4, label="Training data")
    plt.scatter(X_test, y_test, c="r", s=4, label="Testing data")
    plt.title("Simple Linear Regression Dataset")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.legend()
    plt.savefig("regression_data.png")
    plt.close()
    
    return X_train, X_test, y_train, y_test

# 2. Create a model
def create_regression_model(in_features=1, hidden_units=10, out_features=1):
    """Create a neural network for regression."""
    model = nn.Sequential(
        nn.Linear(in_features=in_features, out_features=hidden_units),
        nn.Linear(in_features=hidden_units, out_features=hidden_units),
        nn.Linear(in_features=hidden_units, out_features=out_features)
    ).to(device)
    
    return model

# 3. Train the model
def train_regression_model(model, X_train, y_train, X_test, y_test, epochs=1000, lr=0.01):
    """Train a regression model."""
    # Set random seeds
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    
    # Loss and optimizer
    loss_fn = nn.L1Loss()  # MAE loss for regression data
    optimizer = torch.optim.SGD(params=model.parameters(), lr=lr)
    
    # Put data on target device
    X_train, y_train = X_train.to(device), y_train.to(device)
    X_test, y_test = X_test.to(device), y_test.to(device)
    
    # Training loop
    results = {
        "train_loss": [],
        "test_loss": []
    }
    
    for epoch in range(epochs):
        # Training
        model.train()
        
        # Forward pass
        y_pred = model(X_train)
        
        # Calculate loss
        loss = loss_fn(y_pred, y_train)
        
        # Optimizer zero grad
        optimizer.zero_grad()
        
        # Loss backward
        loss.backward()
        
        # Optimizer step
        optimizer.step()
        
        # Testing
        model.eval()
        with torch.inference_mode():
            test_pred = model(X_test)
            test_loss = loss_fn(test_pred, y_test)
        
        # Store results
        results["train_loss"].append(loss.item())
        results["test_loss"].append(test_loss.item())
        
        # Print progress
        if epoch % 100 == 0:
            print(f"Epoch: {epoch} | Loss: {loss:.5f} | Test Loss: {test_loss:.5f}")
    
    return results

# 4. Make predictions and plot results
def plot_regression_results(model, X_train, y_train, X_test, y_test):
    """Plot regression predictions."""
    # Turn on evaluation mode
    model.eval()
    
    # Make predictions
    with torch.inference_mode():
        y_preds = model(X_test.to(device))
    
    # Plot data and predictions
    # Move tensors back to CPU for plotting with matplotlib
    plot_predictions(
        train_data=X_train.cpu(),
        train_labels=y_train.cpu(),
        test_data=X_test.cpu(),
        test_labels=y_test.cpu(),
        predictions=y_preds.cpu()
    )
    plt.title("Regression Model Predictions")
    plt.savefig("regression_predictions.png")
    plt.close()

def main():
    """Run the regression validation experiment."""
    # 1. Create data
    X_train, X_test, y_train, y_test = create_regression_data()
    
    # 2. Create model
    model = create_regression_model()
    print(f"Model architecture: {model}")
    
    # 3. Train model
    print("\nTraining regression model...")
    results = train_regression_model(model, X_train, y_train, X_test, y_test)
    
    # 4. Plot results
    plot_regression_results(model, X_train, y_train, X_test, y_test)
    
    # 5. Plot loss curves
    plt.figure(figsize=(10, 7))
    plt.plot(results["train_loss"], label="Train Loss")
    plt.plot(results["test_loss"], label="Test Loss")
    plt.title("Loss Curves")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("regression_loss_curves.png")
    plt.close()
    
    print("\nRegression validation complete. Check the generated images for visualization of results.")

if __name__ == "__main__":
    main()
