#!/usr/bin/env julia
# Comparison of uniform vs adaptive sampling strategies

using MutualInformation

println("="^70)
println("Comparing Sampling Strategies")
println("="^70)

# Test function: Localized wavefunction (concentrated amplitude)
# This represents a state with strong nearest-neighbor correlations
function localized_wavefunction(x)
    # High amplitude when most neighbors match, low otherwise
    matches = sum(x[i] == x[i+1] for i in 1:length(x)-1)
    return exp(2.0 * matches)  # Exponentially favor matching neighbors
end

len = 12
n_samples = 50000

println("\nTest function: Localized wavefunction with nearest-neighbor matching")
println("System size: $len sites")
println("Number of samples: $n_samples")
println()

# Test uniform sampling
println("Testing UNIFORM sampling...")
@time MI_uniform = mutualinformation_sampled(
    localized_wavefunction, len;
    n_samples=n_samples,
    sampling_strategy=:uniform
)

# Test adaptive sampling
println("\nTesting ADAPTIVE sampling...")
@time MI_adaptive = mutualinformation_sampled(
    localized_wavefunction, len;
    n_samples=n_samples,
    sampling_strategy=:adaptive
)

println("\n" * "="^70)
println("Results Comparison")
println("="^70)

# Compare nearest-neighbor MI values
println("\nNearest-neighbor mutual information:")
println("Site Pair | Uniform  | Adaptive | Difference")
println("-"^50)
for i in 1:len-1
    diff = MI_adaptive[i, i+1] - MI_uniform[i, i+1]
    @printf("%2d - %2d   | %.4f   | %.4f   | %+.4f\n",
            i, i+1, MI_uniform[i, i+1], MI_adaptive[i, i+1], diff)
end

# Compare next-nearest-neighbor MI values
println("\nNext-nearest-neighbor mutual information (site 1 with others):")
println("Site Pair | Uniform  | Adaptive | Difference")
println("-"^50)
for i in 3:min(6, len)
    diff = MI_adaptive[1, i] - MI_uniform[1, i]
    @printf("1 - %2d    | %.4f   | %.4f   | %+.4f\n",
            i, MI_uniform[1, i], MI_adaptive[1, i], diff)
end

println("\n" * "="^70)
println("Summary")
println("="^70)
println("For localized wavefunctions, adaptive sampling should:")
println("  • Give more consistent results (lower variance)")
println("  • Better capture the correlation structure")
println("  • Be especially effective with fewer samples")
println()
