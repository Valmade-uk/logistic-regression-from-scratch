# Logistic Regression from Scratch — Spaceship Titanic

A machine learning project that implements **logistic regression from scratch using NumPy** and applies it to the Spaceship Titanic binary classification problem.

The notebook covers the complete workflow from exploratory data analysis and feature engineering to forward propagation, backpropagation, gradient descent, prediction, and model evaluation.

## Dataset

This project uses the Spaceship Titanic dataset available on Kaggle:

[Spaceship Titanic — Kaggle Competition](https://www.kaggle.com/competitions/spaceship-titanic)

The dataset is not included in this repository.  
Download the data from Kaggle and place the CSV file inside the `data/` directory before running the notebook.

## Project Overview

The goal is to predict the binary target `Transported` using passenger-related features.

Instead of relying on a ready-made logistic regression estimator, the model training logic is implemented manually with NumPy in order to understand the core mechanics behind a simple neural network:

- parameter initialization
- linear transformation
- sigmoid activation
- binary cross-entropy loss
- forward propagation
- backpropagation
- gradient descent
- prediction and accuracy evaluation

## Workflow

### 1. Exploratory Data Analysis

The notebook explores:

- target distribution
- categorical, continuous, and text features
- correlations between numerical variables
- service usage patterns

Additional binary features are created to indicate whether a passenger used services such as:

- RoomService
- FoodCourt
- ShoppingMall
- Spa
- VRDeck

### 2. Data Preprocessing

The preprocessing pipeline includes:

- median imputation for numerical missing values
- explicit categories for missing categorical values
- missing-value indicators for `CryoSleep` and `VIP`
- conversion of binary variables to integers
- one-hot encoding of `HomePlanet` and `Destination`
- removal of unprocessed text columns
- stratified train/test split
- matrix transposition for the NumPy implementation

### 3. Logistic Regression from Scratch

The model is implemented using NumPy and includes:

```python
z = np.dot(w.T, x_train) + b
```

followed by the sigmoid activation:

```python
y_head = 1 / (1 + np.exp(-z))
```

Binary cross-entropy is used as the loss function.

Gradients for the weights and bias are calculated manually and the parameters are updated using gradient descent.

### 4. Model Evaluation

Training configuration used in the notebook:

- learning rate: `0.00001`
- iterations: `50`

Results:

| Metric | Accuracy |
|---|---:|
| Training accuracy | 77.94% |
| Test accuracy | 79.37% |

The notebook also visualizes the change in the cost function during training.

## Technologies

- Python
- NumPy
- pandas
- Matplotlib
- scikit-learn
- Jupyter Notebook

## Project Structure

```text
logistic-regression-from-scratch/
├── Notebooks/
│   └── logistic_regression_from_scratch.ipynb
├── .gitignore
└── README.md
```

## Installation

Clone the repository and create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the required packages:

```bash
pip install numpy pandas matplotlib scikit-learn jupyter ipykernel
```

Open the notebook in VS Code or Jupyter and run the cells in order.

## What I Learned

This project was created as part of my Deep Learning studies and focuses on understanding what happens inside a simple neural network before moving to deep learning frameworks such as PyTorch.

The main learning outcomes were:

- understanding logistic regression as a single-neuron binary classifier
- implementing forward propagation manually
- understanding sigmoid activation
- calculating binary cross-entropy loss
- implementing gradients and backpropagation
- updating model parameters with gradient descent
- preparing tabular data for model training
- evaluating train and test performance

## Future Improvements

Possible next steps include:

- feature scaling
- experimenting with different learning rates and iteration counts
- comparing the manual implementation with `sklearn.linear_model.LogisticRegression`
- adding validation metrics such as precision, recall, F1-score, ROC-AUC, and a confusion matrix
- refactoring preprocessing into a reusable pipeline

## Author

Valentyn
