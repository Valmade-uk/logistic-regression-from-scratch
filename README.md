# Neural Networks from Scratch to PyTorch

A collection of machine learning and deep learning projects created as part of my studies.

This repository follows a practical progression from implementing the mechanics of a simple neural network manually with NumPy to building and training neural networks with PyTorch.

## Projects

### 1. Logistic Regression from Scratch — Spaceship Titanic

A binary classification project that implements logistic regression manually using NumPy.

**Notebook:**  
`notebooks/logistic_regression_from_scratch/logistic_regression_from_scratch.ipynb`

Main topics:

- exploratory data analysis
- feature engineering
- missing-value handling
- one-hot encoding
- parameter initialization
- sigmoid activation
- binary cross-entropy loss
- forward propagation
- backpropagation
- gradient descent
- prediction and evaluation

Final results:

| Metric | Accuracy |
|---|---:|
| Training accuracy | 77.94% |
| Test accuracy | 79.37% |

---

### 2. Multiclass Classification with PyTorch — Palmer Penguins

A multiclass classification project using a small feed-forward neural network implemented with PyTorch.

**Notebook:**  
`notebooks/multiclass_classification_penguins/multiclass_classification_penguins.ipynb`

The model classifies penguins into three species:

- Adelie
- Gentoo
- Chinstrap

Main topics:

- exploratory data analysis
- numerical feature standardization
- one-hot encoding of categorical features
- train/validation split
- PyTorch tensor preparation
- feed-forward neural networks
- ReLU activation
- Cross-Entropy Loss
- SGD optimization
- training and validation tracking
- loss and accuracy analysis

Final results:

| Metric | Accuracy |
|---|---:|
| Training accuracy | 98.65% |
| Validation accuracy | 97.27% |

## Technologies

- Python
- NumPy
- pandas
- PyTorch
- scikit-learn
- Matplotlib
- seaborn
- Jupyter Notebook

## Repository Structure

```text
neural-networks-from-scratch-to-pytorch/
├── notebooks/
│   ├── logistic_regression_from_scratch/
│   │   ├── logistic_regression_from_scratch.ipynb
│   │   └── README.md
│   └── multiclass_classification_penguins/
│       ├── multiclass_classification_penguins.ipynb
│       └── README.md
├── .gitignore
└── README.md
```

## Datasets

Dataset files are not included in the repository.

### Spaceship Titanic

Available on Kaggle:

[Spaceship Titanic — Kaggle Competition](https://www.kaggle.com/competitions/spaceship-titanic)

### Palmer Penguins

The Palmer Penguins dataset is used for the multiclass classification project.

Place the required CSV files in the local `data/` directory before running the notebooks.

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd <repository-name>
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the main dependencies:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn torch jupyter ipykernel
```

Open the notebooks in VS Code or Jupyter and run the cells in order.

## Learning Progression

The first project focuses on understanding how a simple neural network works internally by implementing logistic regression manually with NumPy.

The second project moves to PyTorch and introduces multiclass classification, automatic differentiation, neural network modules, ReLU activation, Cross-Entropy Loss, and SGD optimization.

Future projects will extend this progression toward deeper neural networks, computer vision, and NLP.

## Author

Valentyn
