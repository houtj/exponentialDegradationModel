# Exponential Degradation Model

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)

A Python implementation of the exponential degradation model for predicting remaining useful life (RUL) of degrading systems. This package provides a probabilistic approach to prognostics using Bayesian parameter updating.

## Overview

This implementation is based on the exponential degradation model described in:

> Gebraeel, N. (2006). "Sensory-Updated Residual Life Distributions for Components With Exponential Degradation Patterns." *IEEE Transactions on Automation Science and Engineering*, 3(4), 382-393.

The model is designed for systems that exhibit exponential degradation patterns and provides:

- **Bayesian parameter updating** as new degradation measurements become available
- **Probabilistic RUL predictions** with confidence intervals
- **Uncertainty quantification** through parameter variance tracking
- **Correlation modeling** between degradation parameters

This package provides functionality similar to MATLAB's [`exponentialDegradationModel`](https://www.mathworks.com/help/predmaint/ref/exponentialdegradationmodel.html) from the Predictive Maintenance Toolbox.

## Features

- ✅ **Bayesian updating**: Refines model parameters with each new observation
- ✅ **Multiple observation fitting**: Efficient batch parameter updates
- ✅ **Comprehensive predictions**: Point estimates, confidence intervals, and full probability distributions
- ✅ **Truncated distributions**: Ensures physically meaningful (positive) RUL predictions
- ✅ **Correlation handling**: Accounts for parameter correlations in uncertainty propagation
- ✅ **Numerical stability**: Robust handling of edge cases and numerical issues

## Installation

### From PyPI (when published)

```bash
pip install exponential-degradation-model
```

### From Source

```bash
git clone https://github.com/houtj/exponentialDegradationModel.git
cd exponentialDegradationModel
pip install -e .
```

### Development Installation

```bash
pip install -e ".[dev]"
```

## Quick Start

```python
import numpy as np
from exponential_degradation import ExponentialDegradationModel

# Create model with failure threshold
model = ExponentialDegradationModel(threshold=10.0)

# Prepare degradation measurements
times = np.array([1, 2, 3, 4, 5])
measurements = np.array([1.5, 2.1, 2.9, 4.2, 5.8])

# Fit the model to observations
model.fit(measurements, times)

# Predict remaining useful life
rul_result = model.predict_rul(confidence_level=0.95)

print(f"Predicted RUL: {rul_result['RUL']:.2f}")
print(f"Mean RUL: {rul_result['mean']:.2f}")
print(f"95% CI: [{rul_result['CI'][0]:.2f}, {rul_result['CI'][1]:.2f}]")
```

## Mathematical Model

### Degradation Path

The model assumes an exponential degradation path:

```
y(t) = exp(θ + β·t) + φ
```

where:
- **θ** (theta): Initial value parameter (log-scale)
- **β** (beta): Growth rate parameter (degradation rate)
- **φ** (phi): Offset parameter
- **t**: Time

### Failure Condition

The system fails when the degradation measurement reaches a threshold **D**:

```
y(t) ≥ D
```

### Remaining Useful Life

The RUL at time *t* is:

```
RUL = L - t
```

where **L** is the time to failure:

```
L = [ln(D - φ) - θ] / β
```

### Bayesian Updating

As new measurements arrive, the model uses Bayesian updating to refine the parameter estimates (θ, β) and their uncertainties (variances and correlation). This provides increasingly accurate RUL predictions as more degradation data becomes available.

### Uncertainty Quantification

The RUL prediction accounts for parameter uncertainty through variance propagation:

```
Var(L) = (1/β²) · [Var(θ) + (μ_L·β)²·Var(β) - 2·μ_L·β·Cov(θ,β)]
```

The RUL distribution is modeled as a truncated normal distribution (truncated at zero) to ensure physically meaningful positive RUL values.

## API Reference

### ExponentialDegradationModel

#### Initialization

```python
ExponentialDegradationModel(
    threshold,
    theta=1.0,
    theta_variance=1e6,
    beta=1.0,
    beta_variance=1e6,
    rho=0.0,
    phi=-1.0
)
```

**Parameters:**

- `threshold` (float): Failure threshold value
- `theta` (float, optional): Initial value parameter. Default: 1.0
- `theta_variance` (float, optional): Initial variance of theta. Default: 1e6 (high uncertainty)
- `beta` (float, optional): Growth rate parameter. Default: 1.0
- `beta_variance` (float, optional): Initial variance of beta. Default: 1e6
- `rho` (float, optional): Correlation coefficient between theta and beta [-1, 1]. Default: 0.0
- `phi` (float, optional): Offset parameter. Default: -1.0

#### Methods

##### `update(measurement, time)`

Update model parameters with a single new observation.

```python
model.update(measurement=2.5, time=1.0)
```

**Parameters:**
- `measurement` (float): Degradation measurement at the given time
- `time` (float): Time point of the measurement

##### `fit(measurements, times)`

Update model parameters with multiple observations (batch update).

```python
measurements = np.array([1.5, 2.1, 2.9, 4.2, 5.8])
times = np.array([1, 2, 3, 4, 5])
model.fit(measurements, times)
```

**Parameters:**
- `measurements` (array-like): Array of degradation measurements
- `times` (array-like): Array of corresponding time points

##### `predict()`

Get a simple point estimate of the mean RUL.

```python
rul = model.predict()
```

**Returns:**
- `float`: Mean remaining useful life

##### `predict_rul(confidence_level=0.95, num_samples=1000)`

Get comprehensive RUL prediction with confidence intervals and probability distributions.

```python
result = model.predict_rul(confidence_level=0.95, num_samples=1000)
```

**Parameters:**
- `confidence_level` (float, optional): Confidence level for CI (0-1). Default: 0.95
- `num_samples` (int, optional): Number of points for PDF/CDF. Default: 1000

**Returns:**

Dictionary containing:
- `'RUL'`: Median RUL (recommended point estimate)
- `'mean'`: Mean RUL
- `'std'`: Standard deviation of RUL
- `'CI'`: Tuple of (lower_bound, upper_bound)
- `'pdf_time'`: Array of time points for PDF
- `'pdf_values'`: PDF values at each time point
- `'cdf_values'`: CDF values at each time point
- `'mu_untruncated'`: Mean of untruncated distribution
- `'sigma_untruncated'`: Std of untruncated distribution

## Usage Examples

### Example 1: Sequential Updating

```python
import numpy as np
from exponential_degradation import ExponentialDegradationModel

# Initialize model
model = ExponentialDegradationModel(threshold=15.0)

# Sequential updates as new measurements arrive
times = [1, 2, 3, 4, 5]
measurements = [2.1, 3.5, 5.2, 7.8, 10.5]

for t, m in zip(times, measurements):
    model.update(m, t)
    rul = model.predict()
    print(f"Time {t}: RUL = {rul:.2f}")
```

### Example 2: Batch Fitting with Visualization

```python
import numpy as np
import matplotlib.pyplot as plt
from exponential_degradation import ExponentialDegradationModel

# Generate synthetic degradation data
np.random.seed(42)
true_theta = 0.5
true_beta = 0.3
times = np.array([0, 1, 2, 3, 4, 5])
measurements = np.exp(true_theta + true_beta * times) + np.random.normal(0, 0.1, len(times))

# Create and fit model
model = ExponentialDegradationModel(threshold=20.0)
model.fit(measurements, times)

# Predict RUL with confidence intervals
result = model.predict_rul(confidence_level=0.95)

# Visualize results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Plot degradation path
ax1.scatter(times, measurements, label='Measurements', color='blue')
ax1.axhline(y=20.0, color='r', linestyle='--', label='Failure Threshold')
ax1.set_xlabel('Time')
ax1.set_ylabel('Degradation')
ax1.set_title('Degradation Path')
ax1.legend()
ax1.grid(True)

# Plot RUL distribution
ax2.plot(result['pdf_time'], result['pdf_values'], label='PDF', color='green')
ax2.axvline(x=result['RUL'], color='b', linestyle='--', label=f"Median RUL: {result['RUL']:.2f}")
ax2.axvline(x=result['CI'][0], color='gray', linestyle=':', label='95% CI')
ax2.axvline(x=result['CI'][1], color='gray', linestyle=':')
ax2.set_xlabel('Remaining Useful Life')
ax2.set_ylabel('Probability Density')
ax2.set_title('RUL Distribution')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()
```

### Example 3: Batch Fitting and Sequential Updating

```python
import numpy as np
from exponential_degradation import ExponentialDegradationModel

# Generate synthetic degradation data (30 points total)
np.random.seed(42)
true_theta = 0.2
true_beta = 0.15
all_times = np.linspace(1, 30, 30)
all_measurements = np.exp(true_theta + true_beta * all_times) + np.random.normal(0, 0.3, 30)

# Split data: first 5 points for batch fitting, remaining 25 for sequential updates
initial_times = all_times[:5]
initial_measurements = all_measurements[:5]
update_times = all_times[5:]
update_measurements = all_measurements[5:]

# Initialize model with appropriate threshold
model = ExponentialDegradationModel(threshold=200.0)

# Step 1: Batch fitting with initial 5 points
print("Step 1: Initial batch fitting with 5 points")
model.fit(initial_measurements, initial_times)
result = model.predict_rul()
print(f"  Current time: {model.ti:.2f}")
print(f"  Predicted RUL: {result['RUL']:.2f}")
print(f"  95% CI: [{result['CI'][0]:.2f}, {result['CI'][1]:.2f}]")
print(f"  Parameters: theta={model.theta:.3f}, beta={model.beta:.3f}")

# Step 2: Sequential updates with remaining 25 points
print("\nStep 2: Sequential updates with remaining 25 points")
print(f"{'Time':<8} {'Measurement':<14} {'RUL':<10} {'CI Width':<12} {'Theta':<10} {'Beta':<10}")
print("-" * 72)

rul_history = []
ci_width_history = []

for t, m in zip(update_times, update_measurements):
    model.update(m, t)
    result = model.predict_rul()
    ci_width = result['CI'][1] - result['CI'][0]
    
    # Store for analysis
    rul_history.append(result['RUL'])
    ci_width_history.append(ci_width)
    
    # Print every 5th update for brevity
    if int(t) % 5 == 0 or t == update_times[-1]:
        print(f"{t:<8.1f} {m:<14.2f} {result['RUL']:<10.2f} {ci_width:<12.2f} "
              f"{model.theta:<10.3f} {model.beta:<10.3f}")

# Final prediction summary
print("\nFinal Prediction (after 30 observations):")
final_result = model.predict_rul()
print(f"  Median RUL: {final_result['RUL']:.2f}")
print(f"  Mean RUL: {final_result['mean']:.2f}")
print(f"  Std Dev: {final_result['std']:.2f}")
print(f"  95% CI: [{final_result['CI'][0]:.2f}, {final_result['CI'][1]:.2f}]")
print(f"  Final parameters: theta={model.theta:.4f}, beta={model.beta:.4f}, rho={model.rho:.4f}")

# Analysis
print("\nObservation: CI width decreased from", f"{ci_width_history[0]:.2f} to {ci_width_history[-1]:.2f}")
print("             Parameter estimates converged with more data")
print(f"\nThis example demonstrates the hybrid approach:")
print(f"  1. Batch fitting (fit) with initial data for fast initialization")
print(f"  2. Sequential updating (update) for real-time monitoring")
print(f"  3. Uncertainty reduction as more observations are incorporated")
```

### Example 4: Comparing Different Initial Parameters

```python
import numpy as np
from exponential_degradation import ExponentialDegradationModel

# Same data, different initial parameter uncertainties
times = np.array([1, 2, 3, 4, 5])
measurements = np.array([1.5, 2.5, 4.0, 6.5, 10.0])

# High initial uncertainty (default)
model_uncertain = ExponentialDegradationModel(
    threshold=15.0,
    theta_variance=1e6,
    beta_variance=1e6
)
model_uncertain.fit(measurements, times)
result_uncertain = model_uncertain.predict_rul()

# Low initial uncertainty (strong prior)
model_informed = ExponentialDegradationModel(
    threshold=15.0,
    theta=0.5,
    theta_variance=0.1,
    beta=0.4,
    beta_variance=0.01
)
model_informed.fit(measurements, times)
result_informed = model_informed.predict_rul()

print("High Uncertainty Prior:")
print(f"  RUL: {result_uncertain['RUL']:.2f}")
print(f"  95% CI: [{result_uncertain['CI'][0]:.2f}, {result_uncertain['CI'][1]:.2f}]")
print(f"  CI Width: {result_uncertain['CI'][1] - result_uncertain['CI'][0]:.2f}")

print("\nLow Uncertainty Prior (Informed):")
print(f"  RUL: {result_informed['RUL']:.2f}")
print(f"  95% CI: [{result_informed['CI'][0]:.2f}, {result_informed['CI'][1]:.2f}]")
print(f"  CI Width: {result_informed['CI'][1] - result_informed['CI'][0]:.2f}")
```

## When to Use This Model

### Suitable Applications

This model is appropriate when:

- ✅ Degradation follows an **exponential growth pattern**
- ✅ Measurements are available over time
- ✅ A clear **failure threshold** can be defined
- ✅ Degradation is **monotonically increasing**
- ✅ **Uncertainty quantification** is important

### Example Applications

- **Bearing degradation**: Vibration amplitude growth
- **Battery capacity fade**: Capacity loss over charge cycles
- **Crack propagation**: Crack size growth in structural components
- **Wear processes**: Tool wear, brake pad wear
- **Corrosion**: Material thickness loss
- **Sensor drift**: Calibration drift in sensors

### Not Suitable For

- ❌ Non-exponential degradation patterns (linear, power-law, etc.)
- ❌ Non-monotonic degradation (with recovery or fluctuations)
- ❌ Systems without clear failure thresholds
- ❌ Sudden failures without gradual degradation

## Technical Details

### Numerical Stability

The implementation includes several features to ensure numerical stability:

- Variance clamping to prevent negative variances
- Truncation at zero for physically meaningful RUL values
- Safe handling of extreme parameter values
- Fallback strategies for edge cases
- Warning flags for numerical issues

### Parameter Initialization

For systems with weak prior knowledge, use large initial variances (default `1e6`) to allow the data to drive parameter estimates. For systems with strong prior knowledge, set smaller variances and appropriate mean values for `theta` and `beta`.

### Measurement Requirements

For reliable parameter estimation:
- Minimum **3-5 measurements** recommended
- Measurements should span a significant portion of the degradation range
- More frequent measurements early in life improve predictions
- Measurements closer to failure threshold provide more information

## Dependencies

- Python ≥ 3.8
- NumPy ≥ 1.20.0
- SciPy ≥ 1.7.0

Optional (for examples and visualization):
- Matplotlib ≥ 3.5.0
- Jupyter ≥ 1.0.0

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## References

1. Gebraeel, N. (2006). "Sensory-Updated Residual Life Distributions for Components With Exponential Degradation Patterns." *IEEE Transactions on Automation Science and Engineering*, 3(4), 382-393.

2. MATLAB Documentation: [exponentialDegradationModel](https://www.mathworks.com/help/predmaint/ref/exponentialdegradationmodel.html)

3. Si, X. S., Wang, W., Hu, C. H., & Zhou, D. H. (2011). "Remaining useful life estimation–a review on the statistical data driven approaches." *European Journal of Operational Research*, 213(1), 1-14.

## Author

**Tianjun HOU**

- GitHub: [@houtj](https://github.com/houtj)

## Acknowledgments

This implementation is based on the theoretical framework developed by Prof. Nagi Gebraeel and draws inspiration from MATLAB's Predictive Maintenance Toolbox.

---

**Keywords**: prognostics, remaining useful life, RUL, degradation modeling, predictive maintenance, reliability engineering, exponential degradation, Bayesian updating

