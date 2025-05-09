"""
Multi-class classification with PyTorch neural networks
"""

import torch
from torch import nn
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from helper_functions import plot_decision_boundary

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Create accuracy function
def accuracy_fn(y_true, y_pred):
    """Calculate accuracy between truth labels and predictions."""
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct/len(y_pred)) * 100
    return acc

# 1. Create multi-class data
def create_blob_data(n_samples=1000, n_features=2, n_classes=4, random_state=42):
    """Create a dataset of blob data using scikit-learn's make_blobs."""
    # Create multi-class data
    X_blob, y_blob = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=n_classes,
        cluster_std=1.5,  # gives the clusters a little shake up
        random_state=random_state
    )
    
    # Turn data into tensors
    X_blob = torch.from_numpy(X_blob).type(torch.float)
    y_blob = torch.from_numpy(y_blob).type(torch.LongTensor)  # use LongTensor for class indices
    
    # Split into train and test
    X_blob_train, X_blob_test, y_blob_train, y_blob_test = train_test_split(
        X_blob,
        y_blob,
        test_size=0.2,
        random_state=random_state
    )
    
    # Print shapes
    print(f"X_blob_train: {X_blob_train.shape}, y_blob_train: {y_blob_train.shape}")
    print(f"X_blob_test: {X_blob_test.shape}, y_blob_test: {y_blob_test.shape}")
    
    # Visualize the data
    plt.figure(figsize=(10, 7))
    plt.scatter(X_blob[:, 0], X_blob[:, 1], c=y_blob, cmap=plt.cm.RdYlBu)
    plt.title("Multi-class Blob Dataset")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.colorbar(label="Class")
    plt.savefig("blob_data.png")
    plt.close()
    
    return X_blob_train, X_blob_test, y_blob_train, y_blob_test

# 2. Build a multi-class classification model
class BlobModel(nn.Module):
    def __init__(self, input_features, output_features, hidden_units=8):
        """Initialize multi-class classification model.
        
        Args:
            input_features (int): Number of input features to the model
            output_features (int): Number of output features (number of classes)
            hidden_units (int): Number of hidden units between layers, default 8
        """
        super().__init__()
        self.linear_layer_stack = nn.Sequential(
            nn.Linear(in_features=input_features, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=output_features)
        )

    def forward(self, x):
        return self.linear_layer_stack(x)

# 3. Train the model
def train_multiclass_model(model, X_train, y_train, X_test, y_test, epochs=500, lr=0.1):
    """Train a multi-class classification model."""
    # Set random seeds
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    
    # Loss and optimizer
    loss_fn = nn.CrossEntropyLoss()  # Loss function for multi-class classification
    optimizer = torch.optim.SGD(params=model.parameters(), lr=lr)
    
    # Put data on target device
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
        # Training
        model.train()
        
        # 1. Forward pass
        y_logits = model(X_train)  # model outputs raw logits
        y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)  # go from logits -> prediction probabilities -> prediction labels
        
        # 2. Calculate loss/accuracy
        loss = loss_fn(y_logits, y_train)
        acc = accuracy_fn(y_true=y_train, y_pred=y_pred)
        
        # 3. Optimizer zero grad
        optimizer.zero_grad()
        
        # 4. Loss backward
        loss.backward()
        
        # 5. Optimizer step
        optimizer.step()
        
        # Testing
        model.eval()
        with torch.inference_mode():
            # 1. Forward pass
            test_logits = model(X_test)
            test_pred = torch.softmax(test_logits, dim=1).argmax(dim=1)
            
            # 2. Calculate loss/accuracy
            test_loss = loss_fn(test_logits, y_test)
            test_acc = accuracy_fn(y_true=y_test, y_pred=test_pred)
        
        # Store results
        results["train_loss"].append(loss.item())
        results["train_acc"].append(acc)
        results["test_loss"].append(test_loss.item())
        results["test_acc"].append(test_acc)
        
        # Print progress
        if epoch % 50 == 0:
            print(f"Epoch: {epoch} | Loss: {loss:.5f} | Accuracy: {acc:.2f}% | Test Loss: {test_loss:.5f} | Test Accuracy: {test_acc:.2f}%")
    
    return results

# 4. Evaluate the model
def evaluate_multiclass_model(model, X_test, y_test):
    """Evaluate the multi-class model and print classification report."""
    model.eval()
    with torch.inference_mode():
        # Get predictions
        y_logits = model(X_test.to(device))
        y_pred_probs = torch.softmax(y_logits, dim=1)
        y_preds = y_pred_probs.argmax(dim=1)
        
        # Calculate accuracy
        test_acc = accuracy_fn(y_true=y_test.to(device), y_pred=y_preds)
        print(f"Test accuracy: {test_acc:.2f}%")
        
        # Return predictions
        return y_preds

# 5. Visualize multi-class predictions
def plot_multiclass_predictions(model, X_train, y_train, X_test, y_test):
    """Plot decision boundaries for multi-class classification."""
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Train Data")
    plot_decision_boundary(model, X_train, y_train)
    plt.subplot(1, 2, 2)
    plt.title("Test Data")
    plot_decision_boundary(model, X_test, y_test)
    plt.savefig("multiclass_boundaries.png")
    plt.close()
    
    # Plot one more with a custom colormap
    plt.figure(figsize=(10, 7))
    plot_decision_boundary(model, X_test, y_test, cmap=plt.cm.RdYlBu)
    plt.title("Multi-class Decision Boundary")
    plt.savefig("multiclass_detailed_boundary.png")
    plt.close()

def main():
    """Run the multi-class classification experiment."""
    # Define parameters
    NUM_CLASSES = 4
    NUM_FEATURES = 2
    
    # 1. Create data
    X_train, X_test, y_train, y_test = create_blob_data(
        n_samples=1000,
        n_features=NUM_FEATURES,
        n_classes=NUM_CLASSES
    )
    
    # 2. Create model
    model = BlobModel(
        input_features=NUM_FEATURES,
        output_features=NUM_CLASSES,
        hidden_units=8
    ).to(device)
    print(f"Model architecture: {model}")
    
    # 3. Train model
    print("\nTraining multi-class model...")
    results = train_multiclass_model(model, X_train, y_train, X_test, y_test)
    
    # 4. Evaluate model
    print("\nEvaluating model...")
    y_preds = evaluate_multiclass_model(model, X_test, y_test)
    
    # 5. Visualize results
    plot_multiclass_predictions(model, X_train, y_train, X_test, y_test)
    
    # 6. Plot loss and accuracy curves
    plt.figure(figsize=(15, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(results["train_loss"], label="Train Loss")
    plt.plot(results["test_loss"], label="Test Loss")
    plt.title("Loss Curves")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(results["train_acc"], label="Train Accuracy")
    plt.plot(results["test_acc"], label="Test Accuracy")
    plt.title("Accuracy Curves")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True)
    
    plt.savefig("multiclass_learning_curves.png")
    plt.close()
    
    print("\nMulti-class classification experiment complete. Check the generated images for visualizations.")

if __name__ == "__main__":
    main()