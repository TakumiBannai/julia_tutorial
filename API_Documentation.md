# Julia Tutorial - API Documentation

## Overview

This Julia tutorial project provides a comprehensive introduction to data science and machine learning using Julia. The project consists of three main notebooks covering basic Julia syntax, linear regression, and supervised learning techniques.

## Project Structure

```
julia_tutorial/
├── 1_Basics.ipynb              # Julia fundamentals and basic operations
├── 2_LinearRegression.ipynb     # Linear regression with GLM
├── 3_SupervisedLearning.ipynb   # Machine learning algorithms
├── data/                        # Dataset files
│   ├── Boston.csv              # Boston housing dataset
│   └── housing.csv             # Housing dataset
├── 0_Setup_julia.md            # Julia installation and setup guide
└── README.md                   # Project overview
```

## Dependencies

### Required Packages

- **Plots.jl**: Data visualization and plotting
- **CSV.jl**: CSV file reading and writing
- **DataFrames.jl**: Data manipulation and analysis
- **GLM.jl**: Generalized Linear Models
- **Statistics.jl**: Statistical functions
- **StatsPlots.jl**: Statistical plotting
- **DecisionTree.jl**: Decision tree and random forest algorithms
- **SVR.jl**: Support Vector Regression
- **Random.jl**: Random number generation and sampling

### Installation

```julia
using Pkg
Pkg.add(["Plots", "CSV", "DataFrames", "GLM", "Statistics", "StatsPlots", "DecisionTree", "SVR", "Random"])
```

## Public APIs and Functions

### 1. Basic Operations (`1_Basics.ipynb`)

#### `add(x, y)`

**Description**: Performs addition of two numbers.

**Parameters**:
- `x`: First number (Integer or Float)
- `y`: Second number (Integer or Float)

**Returns**: Sum of x and y

**Usage**:
```julia
# Basic addition
result = add(3, 5)
println(result)  # Output: 8

# With variables
a = 10
b = 20
sum_result = add(a, b)  # Returns 30
```

#### Data Structures

**Vectors**:
```julia
# Create vector
vec = [1, 2, 3]

# Access elements (1-indexed)
first_element = vec[1]  # Returns 1
```

**Matrices**:
```julia
# Create matrix
mat = [1 2 3; 4 5 6]

# Check dimensions
size(mat)  # Returns (2, 3)

# Access rows
first_row = mat[1, :]  # Returns [1, 2, 3]

# Transpose
transposed = mat'  # or transpose(mat)
```

#### Visualization Functions

**Basic Plotting**:
```julia
using Plots

# Function plotting
plot(x->x^3+x^2+x, 
     xlabel="x", 
     ylabel="y", 
     label="y=x³+x²+x",
     xlims=(-3, 3), 
     ylims=(-10, 10))
```

**Multiple Plot Types**:
```julia
# Bar chart
plot(rand(50), st=:bar, title="bar")

# Pie chart
plot(rand(5), st=:pie, title="pie")

# Scatter plot
x = 0:0.2:10
plot(x, exp.(-x), st=:scatter, title="scatter", label="exp(-x)")

# Histogram
plot(rand(1000), st=:histogram, title="histogram")
```

**3D Visualization**:
```julia
# Data preparation
x = -3:0.2:3
y = x
z = @. exp(-(x^2+y'^2))

# Different 3D plot types
heatmap(z, title="heatmap")
contour(x, y, z, title="contour")
wireframe(x, y, z, title="wireframe")
surface(x, y, z, title="surface")
```

#### Data Analysis Functions

**Statistical Operations**:
```julia
using Statistics

# Basic statistics
mean_val = Statistics.mean(data)
std_val = Statistics.std(data)
median_val = Statistics.median(data)
var_val = Statistics.var(data)
max_val = maximum(data)
min_val = minimum(data)
```

### 2. Linear Regression (`2_LinearRegression.ipynb`)

#### CSV Data Loading

**Usage**:
```julia
using CSV, DataFrames

# Load CSV file
boston = DataFrame(CSV.File("./data/housing.csv"))
# Alternative syntax
boston = CSV.read("./data/housing.csv", DataFrame)
```

#### Linear Regression API

**Model Training**:
```julia
using GLM

# Create and fit linear model
ols = lm(@formula(MEDV ~ RM + LSTAT + PTRATIO), boston)

# Get coefficients
coefficients = coef(ols)

# Make predictions
predictions = predict(ols)
```

**Visualization**:
```julia
# Scatter plot
plot(boston.RM, boston.MEDV, st=:scatter)

# Histogram
plot(boston.MEDV, st=:hist, bin=50)

# Compare predictions vs actual
plot(predictions)
plot!(boston.MEDV)  # Overlay actual values
```

#### Other Regression Models

**Logistic Regression**:
```julia
logit = glm(@formula(Y ~ X), data, Binomial(), ProbitLink())
```

**Poisson Regression**:
```julia
poisson = fit(@formula(Y ~ X), data, Poisson(), LogLink())
```

### 3. Supervised Learning (`3_SupervisedLearning.ipynb`)

#### `normalizer(X)`

**Description**: Normalizes data by subtracting mean and dividing by standard deviation (z-score normalization).

**Parameters**:
- `X`: Input matrix or array to normalize

**Returns**: Normalized data with mean=0 and std=1

**Usage**:
```julia
# Normalize a dataset
X = Matrix(DataFrame(CSV.File("./data/housing.csv")))
X_normalized = normalizer(X)

# Example with custom data
data = [1.0 2.0 3.0; 4.0 5.0 6.0; 7.0 8.0 9.0]
normalized_data = normalizer(data)
```

**Implementation Details**:
- Uses broadcasting operations (`.-` and `./`)
- Computes mean and standard deviation along first dimension
- Handles multiple features simultaneously

#### `train_test_split(data, ratio)`

**Description**: Splits dataset into training and testing sets with random shuffling.

**Parameters**:
- `data`: Input dataset (matrix or array)
- `ratio`: Fraction of data to use for training (0.0 to 1.0)

**Returns**: Tuple of (train_data, test_data)

**Usage**:
```julia
# 80-20 split
train_X, test_X = train_test_split(X, 0.8)

# 70-30 split
train_data, test_data = train_test_split(dataset, 0.7)

# Custom split
train_set, test_set = train_test_split(my_data, 0.6)
```

**Implementation Details**:
- Uses `Random.shuffle()` for random ordering
- Splits based on number of rows
- Maintains data integrity during splitting

#### Random Forest API

**Model Training**:
```julia
using DecisionTree

# Train Random Forest model
model = build_forest(train_X[:,4], train_X[:,1:3])

# Make predictions
predictions = apply_forest(model, test_X[:,1:3])
```

**Parameters**:
- First argument: Target variable (labels)
- Second argument: Feature matrix
- Returns trained forest model

#### Support Vector Regression API

**Model Training**:
```julia
using SVR

# Train SVR model
model = SVR.train(train_X[:,4], permutedims(train_X[:,1:3]))

# Make predictions
predictions = SVR.predict(model, permutedims(test_X[:,1:3]))
```

**Important Notes**:
- Requires transposed feature matrix (`permutedims()`)
- Automatically normalizes dependent variables
- Suitable for regression tasks

## Data Visualization Examples

### Basic Plots

```julia
# Scatter plot for model evaluation
plot(predictions, actual_values, 
     st=:scatter,
     xlabel="Predictions", 
     ylabel="Actual Values",
     title="Model Performance")
```

### Statistical Plots

```julia
using StatsPlots

# Histogram with custom bins
StatsPlots.histogram(data, bins=20)

# Box plot
StatsPlots.boxplot(categories, values)
```

## Complete Workflow Example

```julia
# 1. Load required packages
using CSV, DataFrames, Statistics, Random, DecisionTree, Plots

# 2. Load and preprocess data
data = DataFrame(CSV.File("./data/housing.csv"))
X = Matrix(data)
X_normalized = normalizer(X)

# 3. Split data
train_X, test_X = train_test_split(X_normalized, 0.8)

# 4. Train model
model = build_forest(train_X[:,4], train_X[:,1:3])

# 5. Make predictions
predictions = apply_forest(model, test_X[:,1:3])

# 6. Visualize results
plot(predictions, test_X[:,4], 
     st=:scatter,
     xlabel="Predictions", 
     ylabel="Actual Values")
```

## Error Handling and Best Practices

### Data Loading
```julia
# Check if file exists before loading
if isfile("./data/housing.csv")
    data = CSV.read("./data/housing.csv", DataFrame)
else
    error("Data file not found")
end
```

### Data Validation
```julia
# Check for missing values
if any(ismissing.(data))
    println("Warning: Missing values detected")
end

# Validate data dimensions
if size(data, 1) < 10
    error("Insufficient data for analysis")
end
```

### Memory Management
```julia
# For large datasets, consider using views
train_view = @view data[1:train_size, :]
test_view = @view data[train_size+1:end, :]
```

## Performance Considerations

1. **Data Types**: Use appropriate data types (Float64, Int64) for numerical computations
2. **Broadcasting**: Use vectorized operations with `@.` macro for better performance
3. **Memory**: Consider using `@views` for large datasets to avoid copying
4. **Plotting**: Use `fmt=:png` for complex plots to reduce memory usage

## Troubleshooting

### Common Issues

1. **Package Loading**: Ensure all required packages are installed
2. **Data Paths**: Use relative paths from project root
3. **Matrix Dimensions**: Check data dimensions before operations
4. **Index Errors**: Remember Julia uses 1-based indexing

### Performance Tips

1. Use `@time` macro to measure execution time
2. Profile code with `@profile` for optimization
3. Use in-place operations when possible (functions ending with `!`)
4. Consider using `StaticArrays.jl` for small, fixed-size arrays

## License and Contributing

This tutorial is designed for educational purposes. Feel free to extend and modify the code for your own projects.