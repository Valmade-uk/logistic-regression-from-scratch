# Logistic Regression from Scratch — Spaceship Titanic

A binary classification project that implements **logistic regression from scratch using NumPy** and applies it to the Spaceship Titanic dataset.

The goal of this project is not only to train a classifier, but to understand the mechanics behind a simple neural network before relying on higher-level deep learning frameworks.

## Objective

Predict the binary target `Transported` using passenger-related features.

Instead of using a ready-made estimator, the training process is implemented manually with NumPy.

## Workflow

### Exploratory Data Analysis

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

### Data Preprocessing

The preprocessing workflow includes:

- median imputation for numerical missing values
- explicit categories for missing categorical values
- missing-value indicators for `CryoSleep` and `VIP`
- conversion of binary variables to integers
- one-hot encoding of `HomePlanet` and `Destination`
- removal of unprocessed text columns
- stratified train/test split
- matrix transposition for the NumPy implementation

## Logistic Regression from Scratch

The linear transformation is implemented manually:

```python
z = np.dot(w.T, x_train) + b
```

The sigmoid activation converts the linear output to probabilities:

```python
y_head = 1 / (1 + np.exp(-z))
```

The notebook also implements:

- binary cross-entropy loss
- forward propagation
- gradient calculation
- backpropagation
- gradient descent
- prediction

## Training Configuration

- Learning rate: `0.00001`
- Iterations: `50`

## Results

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

## Dataset

The project uses the Spaceship Titanic dataset available on Kaggle:

[Spaceship Titanic — Kaggle Competition](https://www.kaggle.com/competitions/spaceship-titanic)

The dataset is not included in this repository.

Download the required CSV file and place it in the local `data/` directory before running the notebook.

## What I Learned

This project helped reinforce:

- logistic regression as a single-neuron binary classifier
- parameter initialization
- linear transformations
- sigmoid activation
- binary cross-entropy loss
- forward propagation
- gradient calculation
- backpropagation
- gradient descent
- train/test evaluation

## Future Improvements

Possible extensions include:

- feature scaling
- experimenting with different learning rates
- increasing the number of iterations
- comparison with `sklearn.linear_model.LogisticRegression`
- precision, recall, F1-score, ROC-AUC, and confusion matrix
- reusable preprocessing pipelines

## Author

Valentyn
