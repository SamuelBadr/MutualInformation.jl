#!/usr/bin/env julia

using Pkg
Pkg.activate(".")

using Printf
using LinearAlgebra
import MutualInformation as MI

println("="^70)
println("Comparing Exact vs Sampled Mutual Information Calculation")
println("="^70)

# Test 1: Small system - compare exact vs sampled
println("\n" * "="^70)
println("Test 1: Small system (len=8) - Accuracy validation")
println("="^70)

len_small = 8
f_simple(x) = exp(-sum((xi - 1.5)^2 for xi in x) / 2)

println("\nComputing exact MI matrix (len=$len_small)...")
@time MI_exact = MI.mutualinformation(Float64, f_simple, fill(2, len_small))

println("\nComputing sampled MI matrix (n_samples=50000)...")
@time MI_sampled = MI.mutualinformation_sampled(Float64, f_simple, len_small; n_samples=50000)

println("\n" * "-"^70)
println("Results comparison:")
println("-"^70)

# Compute error metrics
abs_errors = abs.(MI_exact - MI_sampled)
rel_errors = abs_errors ./ (MI_exact .+ 1e-10)  # Avoid division by zero

# Get off-diagonal values only
mask = .!(I(len_small) .== 1)
off_diag_abs_errors = abs_errors[mask]
off_diag_rel_errors = rel_errors[mask]

@printf("Max absolute error: %.6f nats\n", maximum(off_diag_abs_errors))
@printf("Mean absolute error: %.6f nats\n", sum(off_diag_abs_errors) / length(off_diag_abs_errors))
@printf("Max relative error: %.2f%%\n", 100 * maximum(off_diag_rel_errors))
@printf("Mean relative error: %.2f%%\n", 100 * sum(off_diag_rel_errors) / length(off_diag_rel_errors))

println("\nSample comparison (first 4x4 block):")
println("Exact:")
display(MI_exact[1:4, 1:4])
println("\nSampled:")
display(MI_sampled[1:4, 1:4])

# Test 2: Medium system - sampled only
println("\n" * "="^70)
println("Test 2: Medium system (len=15) - Sampled method only")
println("="^70)

len_medium = 15
println("\nComputing sampled MI matrix (len=$len_medium, n_samples=100000)...")
@time MI_medium = MI.mutualinformation_sampled(Float64, f_simple, len_medium; n_samples=100000)

println("\nStatistics:")
@printf("Maximum MI: %.6f nats\n", maximum(MI_medium))
@printf("Mean MI (off-diagonal): %.6f nats\n", sum(MI_medium) / (len_medium^2 - len_medium))
@printf("Min MI (off-diagonal): %.6f nats\n", minimum(MI_medium[MI_medium .> 0]))

# Test 3: Large system - demonstrate scalability
println("\n" * "="^70)
println("Test 3: Large system (len=30) - Demonstration")
println("="^70)
println("(This would be completely infeasible with exact method)")

len_large = 30
println("\nComputing sampled MI matrix (len=$len_large, n_samples=100000)...")
@time MI_large = MI.mutualinformation_sampled(Float64, f_simple, len_large; n_samples=100000)

println("\nStatistics:")
@printf("Maximum MI: %.6f nats\n", maximum(MI_large))
@printf("Mean MI (off-diagonal): %.6f nats\n", sum(MI_large) / (len_large^2 - len_large))

# Show a sample of the matrix
println("\nSample region (sites 1-5):")
display(MI_large[1:5, 1:5])

println("\n" * "="^70)
println("Complexity comparison:")
println("="^70)
println("Exact method for len=30:")
println("  - Would require 2^60 ≈ 10^18 density matrix elements")
println("  - Storage: ~8 exabytes of memory (INFEASIBLE)")
println("\nSampled method for len=30:")
println("  - Uses 100,000 samples")
println("  - Storage: ~24 MB")
println("  - Time: a few seconds")
println("  - Speedup: >10^12 times faster!")
println("="^70)
