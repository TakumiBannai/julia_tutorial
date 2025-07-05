# Julia Tutorial - Data Science and Machine Learning

A comprehensive Julia tutorial covering data science fundamentals, linear regression, and supervised learning techniques.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Quick Start](#quick-start)
- [Documentation](#documentation)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Usage Examples](#usage-examples)
- [Contributing](#contributing)

## 🎯 Overview

This project provides a hands-on introduction to data science and machine learning using Julia. It consists of three progressive notebooks that cover:

1. **Julia Basics** - Core language features and data structures
2. **Linear Regression** - Statistical modeling with GLM
3. **Supervised Learning** - Machine learning algorithms (Random Forest, SVM)

## ✨ Features

- **Complete Julia Setup Guide** - From installation to Jupyter notebook integration
- **Comprehensive Examples** - Real-world datasets and practical applications
- **Data Visualization** - Interactive plots and statistical visualizations
- **Machine Learning Pipeline** - Data preprocessing, model training, and evaluation
- **Best Practices** - Error handling, performance optimization, and memory management

## 🚀 Quick Start

### 1. Install Julia
```bash
# Download from https://julialang.org/downloads/
# Add to PATH (example for macOS)
echo "alias julia='/Applications/Julia-1.6.app/Contents/Resources/julia/bin/julia'" >> .zshrc
```

### 2. Install Required Packages
```julia
using Pkg
Pkg.add(["Plots", "CSV", "DataFrames", "GLM", "Statistics", "StatsPlots", "DecisionTree", "SVR", "Random"])
```

### 3. Set Up Jupyter Notebook
```julia
using Pkg
Pkg.add("IJulia")
```

### 4. Run the Notebooks
```bash
jupyter notebook
# Select Julia kernel and open notebooks in order:
# 1. 1_Basics.ipynb
# 2. 2_LinearRegression.ipynb
# 3. 3_SupervisedLearning.ipynb
```

## 📖 Documentation

📚 **[Complete API Documentation](API_Documentation.md)** - Detailed documentation of all public APIs, functions, and components

⚡ **[Quick Reference Guide](Quick_Reference.md)** - Essential functions and syntax for quick lookup

### Key Components Documented:

#### Public Functions
- `add(x, y)` - Basic arithmetic operations
- `normalizer(X)` - Data normalization with z-score standardization
- `train_test_split(data, ratio)` - Random data splitting for ML workflows

#### Core APIs
- **Data Loading**: CSV reading and DataFrame manipulation
- **Statistical Analysis**: Descriptive statistics and data exploration
- **Linear Regression**: GLM-based modeling with multiple regression types
- **Machine Learning**: Random Forest and Support Vector Regression
- **Visualization**: Comprehensive plotting and statistical charts

#### Advanced Features
- **3D Visualization**: Surface plots, heatmaps, and contour plots
- **Data Preprocessing**: Normalization, missing value handling
- **Model Evaluation**: Prediction visualization and performance metrics
- **Error Handling**: Robust data validation and error management

## 📁 Project Structure

```
julia_tutorial/
├── 1_Basics.ipynb              # Julia fundamentals and basic operations
├── 2_LinearRegression.ipynb     # Linear regression with GLM
├── 3_SupervisedLearning.ipynb   # Machine learning algorithms
├── data/                        # Dataset files
│   ├── Boston.csv              # Boston housing dataset
│   └── housing.csv             # Housing dataset
├── 0_Setup_julia.md            # Julia installation and setup guide
├── API_Documentation.md        # Complete API documentation
├── Quick_Reference.md          # Quick reference guide
└── README.md                   # This file
```

## 🔧 Requirements

### System Requirements
- Julia 1.6+ (recommended)
- Jupyter Notebook or JupyterLab
- 4GB+ RAM for machine learning examples

### Julia Packages
- **Core**: `Plots`, `CSV`, `DataFrames`, `Statistics`
- **Machine Learning**: `GLM`, `DecisionTree`, `SVR`
- **Visualization**: `StatsPlots`
- **Utilities**: `Random`

## 💡 Usage Examples

### Basic Data Analysis
```julia
using CSV, DataFrames, Statistics

# Load data
data = CSV.read("./data/housing.csv", DataFrame)

# Basic statistics
mean_price = mean(data.MEDV)
price_std = std(data.MEDV)
```

### Linear Regression
```julia
using GLM

# Fit model
model = lm(@formula(MEDV ~ RM + LSTAT + PTRATIO), data)

# Make predictions
predictions = predict(model)
```

### Machine Learning Pipeline
```julia
using DecisionTree, Random

# Preprocess data
X = Matrix(data)
X_norm = normalizer(X)

# Split data
train_X, test_X = train_test_split(X_norm, 0.8)

# Train model
model = build_forest(train_X[:,4], train_X[:,1:3])

# Evaluate
predictions = apply_forest(model, test_X[:,1:3])
```

## 🎓 Learning Path

1. **Start with Setup** - Follow `0_Setup_julia.md` for environment setup
2. **Master Basics** - Work through `1_Basics.ipynb` for Julia fundamentals
3. **Explore Statistics** - Use `2_LinearRegression.ipynb` for statistical modeling
4. **Build ML Models** - Complete `3_SupervisedLearning.ipynb` for machine learning
5. **Reference Documentation** - Use `API_Documentation.md` for detailed API reference

## 🤝 Contributing

This tutorial is designed for educational purposes. Contributions are welcome:

1. **Bug Reports** - Report issues with examples or documentation
2. **Feature Requests** - Suggest new topics or improvements
3. **Code Examples** - Add new examples or use cases
4. **Documentation** - Improve clarity and add missing details

## 📜 License

This project is open source and available under the MIT License.

---

**Happy Learning with Julia! 🚀**