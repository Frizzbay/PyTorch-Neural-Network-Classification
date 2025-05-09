# PyTorch Neural Network Classification

This repository contains my exploration of neural network classification with PyTorch. It documents my learning journey implementing various classification models, from simple to complex architectures.

## What I Learned

- Building neural networks using PyTorch's nn.Module and nn.Sequential
- Implementing binary and multi-class classification models
- Understanding the importance of non-linear activation functions
- Creating and manipulating datasets with scikit-learn and PyTorch
- Visualizing decision boundaries and model performance
- Training loops and optimization techniques

## Repository Structure

- `binary_classification.py`: Implementation of models for the circles dataset (binary classification)
- `regression_validation.py`: Linear regression test to validate model architecture
- `multiclass_classification.py`: Implementation for the blob dataset (multi-class classification)
- `helper_functions.py`: Visualization utilities for plotting decision boundaries

## Key Concepts Explored

### Non-linearity in Neural Networks

This project demonstrates why non-linear activation functions (like ReLU) are essential for solving complex problems that aren't linearly separable. The circles dataset is an excellent example where models without non-linear functions fail, while models with them succeed.

### Model Architecture Progression

The repository shows a progression of model complexity:
1. Simple two-layer network
2. Deeper three-layer network
3. Deep network with non-linear activations
4. Multi-class classification model

## Technologies Used

- PyTorch
- scikit-learn
- Matplotlib
- NumPy
- Pandas

## Requirements

```
torch>=2.0.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
numpy>=1.20.0
pandas>=1.3.0
```

## Usage

Each Python file can be run independently to see the different models in action:

```bash
python binary_classification.py
python regression_validation.py
python multiclass_classification.py
```