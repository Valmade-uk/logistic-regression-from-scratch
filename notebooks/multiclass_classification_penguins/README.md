# Multiclass Classification with PyTorch — Palmer Penguins

A multiclass classification project that uses a small feed-forward neural network in PyTorch to classify Palmer Penguins into three species:

- Adelie
- Gentoo
- Chinstrap

## Objective

Build and train a neural network that predicts penguin species using both numerical physical measurements and categorical features.

## Features

### Numerical features

- `bill_length_mm`
- `bill_depth_mm`
- `flipper_length_mm`
- `body_mass_g`

### Categorical features

- `island`
- `sex`

## Data Preparation

The preprocessing workflow includes:

- removing rows with missing values
- encoding the target classes as integer labels
- stratified train/validation split
- fitting preprocessing only on training data
- standardizing numerical features with `StandardScaler`
- one-hot encoding categorical features with `OneHotEncoder`
- converting processed NumPy arrays to PyTorch tensors

The target labels are encoded as:

```text
Adelie     → 0
Gentoo     → 1
Chinstrap  → 2
```

## Model Architecture

The network is intentionally small because the dataset is relatively simple.

```text
Input features
      ↓
Linear layer
      ↓
ReLU
      ↓
Linear layer
      ↓
3 output logits
```

PyTorch implementation:

```python
class LinearModel(torch.nn.Module):

    def __init__(self, in_dim, hidden_dim=20, out_dim=3):
        super().__init__()

        self.features = torch.nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.features(x)
```

The model outputs raw logits. `nn.CrossEntropyLoss()` is used directly on these logits.

## Training

Training configuration:

- Hidden layer: `20` neurons
- Output classes: `3`
- Loss function: `CrossEntropyLoss`
- Optimizer: `SGD`
- Learning rate: `0.01`
- Epochs: `400`

During each epoch the notebook:

- switches the model to training mode
- performs a forward pass
- calculates the loss
- clears previous gradients
- performs backpropagation
- updates model parameters
- calculates training accuracy
- switches to evaluation mode
- calculates validation loss and accuracy without gradients

## Results

Final performance:

| Metric | Accuracy |
|---|---:|
| Training accuracy | 98.65% |
| Validation accuracy | 97.27% |

Training and validation performance remain close, suggesting good generalization without substantial overfitting.

The notebook also visualizes:

- training vs validation loss
- training vs validation accuracy

## Technologies

- Python
- pandas
- NumPy
- PyTorch
- scikit-learn
- Matplotlib
- seaborn
- Jupyter Notebook

## Dataset

This project uses the Palmer Penguins dataset.

The dataset file is not included in this repository. Place the required CSV file in the local `data/` directory before running the notebook.

## What I Learned

This project demonstrates the transition from manual NumPy implementations to PyTorch.

Key concepts include:

- multiclass classification
- feature scaling
- one-hot encoding
- train/validation separation
- `torch.Tensor`
- `nn.Module`
- `nn.Linear`
- ReLU activation
- logits
- Cross-Entropy Loss
- automatic differentiation
- SGD optimization
- `model.train()`
- `model.eval()`
- `torch.no_grad()`
- interpreting loss and accuracy curves

## Future Improvements

Possible extensions include:

- confusion matrix
- per-class precision, recall, and F1-score
- experimenting with hidden-layer sizes
- comparing SGD with Adam
- adding early stopping
- comparing the neural network with classical classifiers
- testing deeper architectures

## Author

Valentyn
