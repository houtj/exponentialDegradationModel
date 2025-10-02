# Quick Reference Guide

## Installation

```bash
pip install exponential-degradation-model
```

## Basic Usage

```python
from exponential_degradation import ExponentialDegradationModel
import numpy as np

# Initialize model
model = ExponentialDegradationModel(threshold=10.0)

# Fit with data
times = np.array([1, 2, 3, 4, 5])
measurements = np.array([1.5, 2.1, 2.9, 4.2, 5.8])
model.fit(measurements, times)

# Predict RUL
result = model.predict_rul()
print(f"RUL: {result['RUL']:.2f}")
```

## Class: ExponentialDegradationModel

### Constructor

```python
ExponentialDegradationModel(
    threshold,              # Required: failure threshold
    theta=1.0,             # Initial value parameter
    theta_variance=1e6,    # Theta uncertainty (large = unknown)
    beta=1.0,              # Growth rate parameter
    beta_variance=1e6,     # Beta uncertainty
    rho=0.0,               # Correlation [-1, 1]
    phi=-1.0               # Offset parameter
)
```

### Methods

#### `fit(measurements, times)`
Batch update with multiple observations.

**Parameters:**
- `measurements` (array): Degradation values
- `times` (array): Time points

**Example:**
```python
times = np.array([1, 2, 3])
measurements = np.array([1.5, 2.5, 3.8])
model.fit(measurements, times)
```

#### `update(measurement, time)`
Sequential update with single observation.

**Parameters:**
- `measurement` (float): Single degradation value
- `time` (float): Time point

**Example:**
```python
model.update(measurement=2.5, time=1.0)
```

#### `predict()`
Simple point estimate of RUL.

**Returns:** float (mean RUL)

**Example:**
```python
rul = model.predict()
print(f"RUL: {rul:.2f}")
```

#### `predict_rul(confidence_level=0.95, num_samples=1000)`
Comprehensive RUL prediction with confidence intervals.

**Parameters:**
- `confidence_level` (float): CI level (0-1), default 0.95
- `num_samples` (int): PDF points, default 1000

**Returns:** dict with keys:
- `'RUL'`: Median RUL (recommended)
- `'mean'`: Mean RUL
- `'std'`: Standard deviation
- `'CI'`: (lower, upper) confidence bounds
- `'pdf_time'`: Time points for PDF
- `'pdf_values'`: PDF values
- `'cdf_values'`: CDF values

**Example:**
```python
result = model.predict_rul(confidence_level=0.95)
print(f"Median RUL: {result['RUL']:.2f}")
print(f"95% CI: [{result['CI'][0]:.2f}, {result['CI'][1]:.2f}]")
```

## Common Patterns

### Pattern 1: Batch Processing
```python
# Initialize model
model = ExponentialDegradationModel(threshold=15.0)

# Fit all data at once
model.fit(all_measurements, all_times)

# Get prediction
result = model.predict_rul()
```

### Pattern 2: Sequential Updates
```python
# Initialize model
model = ExponentialDegradationModel(threshold=15.0)

# Update as new data arrives
for t, m in zip(times, measurements):
    model.update(m, t)
    rul = model.predict()
    print(f"Time {t}: RUL = {rul:.2f}")
```

### Pattern 3: With Prior Knowledge
```python
# Use informed priors (low variance)
model = ExponentialDegradationModel(
    threshold=15.0,
    theta=0.5,           # Known initial value
    theta_variance=0.1,  # Low uncertainty
    beta=0.3,            # Known growth rate
    beta_variance=0.01   # Low uncertainty
)
```

### Pattern 4: Plotting Results
```python
import matplotlib.pyplot as plt

result = model.predict_rul()

# Plot PDF
plt.plot(result['pdf_time'], result['pdf_values'])
plt.axvline(result['RUL'], color='r', linestyle='--', 
            label=f"RUL: {result['RUL']:.2f}")
plt.xlabel('Time')
plt.ylabel('Probability Density')
plt.legend()
plt.show()
```

## Parameter Guidelines

### `threshold`
- **Required**: System failure level
- Set based on: specifications, safety limits, or historical data
- Must be greater than current measurements

### `theta` and `beta`
- **Default**: 1.0 for both (neutral starting point)
- Set based on: previous systems, physics-based models, or expert knowledge
- `theta`: controls initial degradation level
- `beta`: controls degradation rate

### `theta_variance` and `beta_variance`
- **Default**: 1e6 (very uncertain, let data dominate)
- **High values (≥1e4)**: Weak prior, data-driven estimation
- **Low values (≤1.0)**: Strong prior, expert knowledge
- Use high values when uncertain about parameters

### `rho`
- **Default**: 0.0 (no correlation)
- **Range**: -1 to 1
- Usually starts at 0, updates automatically with data
- Negative values common (faster growth → sooner failure)

### `phi`
- **Default**: -1.0 (small offset)
- Represents: measurement bias or baseline offset
- Usually not critical to tune

## Troubleshooting

### Issue: Negative RUL
**Cause:** System already past expected failure time
**Solution:** 
- Check threshold is appropriate
- Verify measurements are correct
- Consider if model assumptions fit data

### Issue: Very wide confidence intervals
**Cause:** High parameter uncertainty
**Solution:**
- Collect more measurements
- Use informed priors (lower variances)
- Check if exponential model fits data

### Issue: RUL not updating with new data
**Cause:** Parameters already converged
**Solution:**
- Normal behavior if estimates are stable
- Check if new data follows expected pattern

### Issue: Import error
**Solution:**
```bash
pip install --upgrade exponential-degradation-model
```

## Performance Tips

1. **Batch vs Sequential**: Use `fit()` for batch data (faster), `update()` for streaming
2. **Number of samples**: Default 1000 is usually sufficient, reduce for speed
3. **Measurement frequency**: More measurements early → better parameter estimation
4. **Data quality**: Remove outliers, ensure monotonic degradation

## Model Assumptions

✅ Works well when:
- Exponential degradation pattern
- Monotonically increasing degradation
- Clear failure threshold
- Gaussian measurement noise

❌ May not work if:
- Non-exponential pattern (linear, power-law, etc.)
- Non-monotonic degradation (with recovery)
- Sudden failures without gradual degradation

## Getting Help

- **Documentation**: See `README.md`
- **Examples**: Run `example.py`
- **Issues**: https://github.com/houtj/exponentialDegradationModel/issues
- **Source**: https://github.com/houtj/exponentialDegradationModel

## References

Gebraeel, N. (2006). "Sensory-Updated Residual Life Distributions for Components 
With Exponential Degradation Patterns." IEEE Transactions on Automation Science 
and Engineering, 3(4), 382-393.

