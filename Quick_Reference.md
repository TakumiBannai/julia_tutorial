# Julia Tutorial - Quick Reference Guide

## 🚀 Essential Functions

### Data Manipulation
```julia
# Load data
data = CSV.read("file.csv", DataFrame)

# Basic info
size(data)           # Dimensions
describe(data)       # Summary statistics
names(data)          # Column names

# Access columns
data.column_name     # By name
data[:, 1]          # By index
data[1:10, :]       # First 10 rows
```

### Statistics
```julia
using Statistics

mean(data)          # Mean
std(data)           # Standard deviation
median(data)        # Median
var(data)           # Variance
minimum(data)       # Minimum value
maximum(data)       # Maximum value
```

### Custom Functions
```julia
# Data normalization
function normalizer(X)
    m = mean(X, dims=1)
    s = std(X, dims=1)
    return (X .- m)./s
end

# Train-test split
function train_test_split(data, ratio)
    n = size(data)[1]
    idx = shuffle(1:n)
    train = data[1:Int(round(n * ratio)), :]
    test = data[Int(round(n * ratio))+1:n, :]
    return train, test
end
```

## 📊 Visualization

### Basic Plots
```julia
using Plots

# Scatter plot
plot(x, y, st=:scatter)

# Line plot
plot(x, y, st=:line)

# Histogram
plot(data, st=:histogram, bins=20)

# Bar chart
plot(x, y, st=:bar)
```

### Advanced Plots
```julia
# Multiple plots
p1 = plot(x, y, title="Plot 1")
p2 = plot(x, z, title="Plot 2")
plot(p1, p2, layout=(1,2))

# 3D plots
heatmap(matrix)
contour(x, y, z)
surface(x, y, z)
```

## 🔬 Machine Learning

### Linear Regression
```julia
using GLM

# Fit model
model = lm(@formula(y ~ x1 + x2), data)

# Get results
coef(model)         # Coefficients
predict(model)      # Predictions
```

### Random Forest
```julia
using DecisionTree

# Train model
model = build_forest(y, X)

# Make predictions
pred = apply_forest(model, X_test)
```

### Support Vector Regression
```julia
using SVR

# Train model
model = SVR.train(y, X')

# Make predictions
pred = SVR.predict(model, X_test')
```

## 🛠️ Data Processing

### Array Operations
```julia
# Create arrays
vec = [1, 2, 3]
mat = [1 2; 3 4]

# Broadcasting
vec .+ 1            # Add 1 to each element
mat .* 2            # Multiply each element by 2

# Matrix operations
mat'                # Transpose
size(mat)           # Dimensions
```

### Control Flow
```julia
# For loops
for i in 1:10
    println(i)
end

# If statements
if condition
    # do something
elseif other_condition
    # do something else
else
    # default action
end
```

## 📦 Package Management

### Install Packages
```julia
using Pkg

# Single package
Pkg.add("PackageName")

# Multiple packages
Pkg.add(["Package1", "Package2"])

# Check status
Pkg.status()
```

### Load Packages
```julia
using CSV              # Full import
using DataFrames      # Full import
import Statistics: mean, std  # Selective import
```

## 🔧 Common Patterns

### Data Pipeline
```julia
# 1. Load data
data = CSV.read("data.csv", DataFrame)

# 2. Preprocess
X = Matrix(data)
X_norm = normalizer(X)

# 3. Split
train_X, test_X = train_test_split(X_norm, 0.8)

# 4. Train model
model = build_forest(train_X[:,end], train_X[:,1:end-1])

# 5. Evaluate
pred = apply_forest(model, test_X[:,1:end-1])
```

### Error Handling
```julia
# Check file exists
if isfile("data.csv")
    data = CSV.read("data.csv", DataFrame)
else
    error("File not found")
end

# Handle missing values
if any(ismissing.(data))
    println("Warning: Missing values detected")
end
```

## 🎯 Performance Tips

### Memory Optimization
```julia
# Use views instead of copies
train_view = @view data[1:train_size, :]

# In-place operations
sort!(array)        # Sorts in-place
```

### Broadcasting
```julia
# Use dot syntax for element-wise operations
result = @. sqrt(x^2 + y^2)

# Instead of loops
for i in 1:length(x)
    result[i] = sqrt(x[i]^2 + y[i]^2)
end
```

## 🐛 Debugging

### Timing and Profiling
```julia
# Time execution
@time expensive_function()

# Benchmark
using BenchmarkTools
@benchmark function_call()
```

### Common Issues
- **1-based indexing**: Julia arrays start at 1, not 0
- **Column-major order**: Julia stores matrices column-wise
- **Broadcasting**: Use `.` for element-wise operations
- **Package conflicts**: Use qualified names (`Package.function`)

## 📚 Quick Help

### Getting Help
```julia
# In Julia REPL
? function_name     # Help mode
```

### Useful Functions
```julia
typeof(x)           # Get type
size(x)             # Get dimensions
length(x)           # Get length
eltype(x)           # Get element type
ndims(x)            # Get number of dimensions
```

---

💡 **Tip**: Keep this reference handy while working through the notebooks!

📖 **For detailed explanations, see [API_Documentation.md](API_Documentation.md)**