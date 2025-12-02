# Sampling-Based Mutual Information Calculation

## Overview

For systems with `len ≥ 15`, the exact mutual information calculation becomes computationally infeasible due to exponential scaling (O(4^len) space complexity). This package provides a Monte Carlo sampling-based approximation that scales linearly with the number of samples.

## When to Use Each Method

### Use Exact Method (`mutualinformation`)
- **Small systems**: `len < 15`
- **High precision required**: Need exact values to machine precision
- **Sparse, highly entangled states**: Bell states, GHZ states, W-states
- **When computation time is not a constraint**

**Complexity**:
- Time: O(len³ × 4^len)
- Space: O(4^len)

### Use Sampled Method (`mutualinformation_sampled`)
- **Large systems**: `len ≥ 15` (required for `len ≥ 30`)
- **Smooth distributions**: Functions in quantics representation, smooth wave functions
- **Acceptable approximation**: 10-20% relative error is sufficient
- **When exact method is infeasible**

**Complexity**:
- Time: O(len² × n_samples)
- Space: O(len × n_samples)

## Usage

```julia
using MutualInformation

# For small systems (exact)
f(x) = exp(-sum((x[i] - 1.5)^2 for i in eachindex(x)))
MI_exact = mutualinformation(Float64, f, fill(2, 10))

# For large systems (sampled approximation)
MI_approx = mutualinformation_sampled(Float64, f, 30; n_samples=100000)
```

## Accuracy Guidelines

The sampling method uses **uniform sampling** of configurations, which works best for smooth, non-sparse distributions.

### Sample Size Recommendations

| System Size (len) | Recommended n_samples | Expected Accuracy | Time (approx)  |
|-------------------|-----------------------|-------------------|----------------|
| 15-20             | 50,000                | 10-15% rel error  | <1 second      |
| 20-30             | 100,000               | 10-20% rel error  | 1-5 seconds    |
| 30-50             | 200,000               | 15-25% rel error  | 5-20 seconds   |
| 50+               | 500,000+              | 20-30% rel error  | 30+ seconds    |

### Distribution Types and Performance

**✓ Works Well:**
- Smooth functions in quantics representation
- Gaussian-like distributions
- Functions with many non-zero configurations
- Nearest-neighbor correlation structures

**⚠ May Require More Samples:**
- Highly peaked distributions
- Sparse quantum states (few non-zero amplitudes)
- Functions with strong localization

**✗ Not Recommended (use exact method):**
- Discrete entangled states (Bell, GHZ, W-states) with small `len`
- Distributions with < 100 non-zero configurations
- When you need exact values

## Performance Example

For a smooth function with `len=30`:

```
Exact method:
- Memory required: ~8 exabytes (INFEASIBLE)
- Time: Would take years

Sampled method (n_samples=100000):
- Memory required: ~24 MB
- Time: ~0.4 seconds
- Speedup: >10^12 times faster!
```

## Advanced: Improving Accuracy

If you need better accuracy for sparse distributions, consider:

1. **Increase sample count**: More samples → better statistics
2. **Importance sampling**: Implement custom sampling from |f(x)|² distribution (future work)
3. **Hybrid approach**: Use exact method for small critical subsystems, sampling for the rest

## Technical Details

The sampled method:
1. Samples configurations `x` uniformly from {1,2}^len
2. Computes weights `w_i = |f(x_i)|²` and normalizes
3. Estimates marginal distributions P(x_A), P(x_B) and joint P(x_A, x_B)
4. Computes Shannon entropies from empirical distributions
5. Calculates MI as I(A:B) = H(A) + H(B) - H(A,B)

**Note**: For large `len`, the method computes classical Shannon entropy from the measurement probability distribution, which approximates the quantum mutual information for diagonal-in-computational-basis measurements.

## Examples

See:
- `scripts/compare_methods.jl` - Comparison of exact vs sampled methods
- `scripts/test_entangled.jl` - Tests with various entanglement structures
- `scripts/plot_mutual_info.jl` - Visualization (adapt for sampled method)
