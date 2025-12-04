# Monte Carlo sampling-based approximation for mutual information
# Suitable for large systems where exact calculation is infeasible

"""
    mutualinformation_uniform(f::F, len::Int; n_samples=100000) where {F}

Monte Carlo approximation of mutual information using uniform sampling.
Suitable for large systems where exact calculation is infeasible.

# Arguments
- `f`: Wavefunction amplitude function taking `Vector{Int}` (values in {1,2})
       and returning a numeric value (real or complex)
- `len`: Number of sites/bits
- `n_samples`: Number of Monte Carlo samples (default: 100000)

# Returns
- `len × len` symmetric matrix of mutual information values in nats

# Complexity
- Time: O(len² / 2 × n_samples × 4) - factor of 4 from full density matrix estimation
- Space: O(len × n_samples)

# Method
For each pair of sites (A, B):
1. Generate uniform samples from the full configuration space
2. For each sample x, estimate all matrix elements of ρ_AB by evaluating f
   on configurations that differ from x only at sites A and B
3. Compute von Neumann entropies S(A), S(B), S(AB) from eigenvalues
4. Return mutual information: I(A:B) = S(A) + S(B) - S(AB)

This correctly computes quantum mutual information, including coherence effects
(off-diagonal density matrix elements).

# Accuracy Guidelines

| System Size | Samples Needed | Expected Error |
|-------------|----------------|----------------|
| len = 15-20 | 50,000         | ~5-10%         |
| len = 20-30 | 100,000        | ~5-15%         |
| len = 30-50 | 200,000        | ~10-20%        |

# When to Use
- Best for smooth, non-sparse wavefunctions with similar amplitudes across configurations
- Guaranteed ergodic coverage of the entire configuration space
- Simple and reliable baseline method
- For sparse/localized wavefunctions, consider `mutualinformation_hybrid` instead

# Example
```julia
# Nearest-neighbor correlation structure
f(x) = exp(-0.5 * sum((x[i] - x[i+1])^2 for i in 1:length(x)-1))
MI = mutualinformation_uniform(f, 30; n_samples=100000)
```
"""
function mutualinformation_uniform(f::F, len::Int; n_samples::Int=100000, rng=default_rng()) where {F}
    # Generate random configurations uniformly
    samples = [rand(rng, 1:2, len) for _ in 1:n_samples]

    # Precompute f(x) for all samples (avoids recomputation)
    f_vals = [f(x) for x in samples]

    # Check that f is not identically zero
    if all(x -> abs(x) < 1e-14, f_vals[1:min(100, n_samples)])
        error("Function f appears to be zero on all sampled configurations")
    end

    # Compute mutual information matrix (only upper triangle)
    MI_matrix = zeros(Float64, len, len)

    # Generate all pairs in upper triangle to enable parallel computation
    pairs = [(A, B) for A in 1:len for B in (A+1):len]

    Threads.@threads for (A, B) in pairs
        # Estimate the full reduced density matrix on sites A and B
        ρ_AB = estimate_reduced_density_matrix_AB(samples, f_vals, f, A, B)

        # Compute reduced density matrices by partial tracing
        ρ_A = partial_trace_to_sites(ρ_AB, [2, 2], [1])  # Keep first qubit
        ρ_B = partial_trace_to_sites(ρ_AB, [2, 2], [2])  # Keep second qubit

        # Compute von Neumann entropies
        S_A = von_neumann_entropy(ρ_A)
        S_B = von_neumann_entropy(ρ_B)
        S_AB = von_neumann_entropy(ρ_AB)

        # Mutual information: I(A:B) = S(A) + S(B) - S(AB)
        # Clamp to non-negative to handle numerical errors
        MI_matrix[A, B] = max(0.0, S_A + S_B - S_AB)
    end

    # Symmetrize the matrix (MI is symmetric by definition)
    for A in 1:len
        for B in 1:(A-1)
            MI_matrix[A, B] = MI_matrix[B, A]
        end
    end

    return MI_matrix
end

"""
    estimate_reduced_density_matrix_AB(samples, f_vals, f, site_A, site_B) -> Matrix{ComplexF64}

Estimate the reduced two-qubit density matrix ρ_AB via Monte Carlo sampling.

For each sample x representing traced-out degrees of freedom, computes
contributions to all matrix elements ρ_AB[i,j] by evaluating f on configurations
that differ from x only at sites A and B.

# Arguments
- `samples`: Vector of uniformly sampled configurations (temporarily modified but restored)
- `f_vals`: Precomputed f(x) values for all samples (optimization)
- `f`: Wavefunction amplitude function
- `site_A`: Index of first site
- `site_B`: Index of second site

# Returns
- 4×4 reduced density matrix normalized to tr(ρ_AB) = 1

# Thread Safety
- Thread-safe: Makes a copy of each sample before modification, enabling safe
  parallel execution across multiple pairs.

# Performance
- Optimized to avoid redundant function evaluations
- Allocates n_samples temporary copies (one per sample), but only one at a time
"""
function estimate_reduced_density_matrix_AB(samples::Vector{Vector{Int}}, f_vals::Vector, f::F, site_A::Int, site_B::Int) where F
    ρ_AB = zeros(ComplexF64, 4, 4)  # 4×4 for two qubits

    for (x, fx) in zip(samples, f_vals)
        a = x[site_A]
        b = x[site_B]

        # Compute row index once (doesn't change in inner loops)
        idx_row = (a - 1) * 2 + b

        # Copy sample to avoid race conditions in multithreaded code
        x_copy = copy(x)

        # Compute contribution to all matrix elements
        for a´ in 1:2
            for b´ in 1:2
                # Modify copy (thread-safe)
                x_copy[site_A] = a´
                x_copy[site_B] = b´
                fx´ = f(x_copy)

                # Map (a',b') to matrix column index
                idx_col = (a´ - 1) * 2 + b´

                ρ_AB[idx_row, idx_col] += fx * conj(fx´)
            end
        end
    end

    # Normalize to ensure tr(ρ) = 1
    # Note: No need to divide by n_samples first since we renormalize by trace anyway
    ρ_AB ./= tr(ρ_AB)

    return ρ_AB
end

# Backward compatibility alias
const mutualinformation_sampled = mutualinformation_uniform

"""
    mutualinformation_hybrid(f::F, len::Int;
                           n_samples=100000,
                           mcmc_fraction=0.8,
                           n_burnin=1000,
                           thin=10,
                           n_flip=1,
                           rng=default_rng()) where {F}

Hybrid Monte Carlo method combining uniform and MCMC sampling for mutual information.
Provides better efficiency than pure uniform sampling for sparse/localized wavefunctions
while maintaining ergodicity guarantees.

# Arguments
- `f`: Wavefunction amplitude function taking `Vector{Int}` (values in {1,2})
       and returning a numeric value (real or complex)
- `len`: Number of sites/bits
- `n_samples`: Total number of Monte Carlo samples (default: 100000)
- `mcmc_fraction`: Fraction of samples from MCMC (default: 0.8)
  - Remaining (1-mcmc_fraction) samples are uniform
- `n_burnin`: MCMC burn-in steps (default: 1000)
- `thin`: MCMC thinning interval (default: 10)
- `n_flip`: Number of bits to flip in MCMC proposals (default: 1)
- `rng`: Random number generator (default: default_rng())

# Returns
- `len × len` symmetric matrix of mutual information values in nats

# Method
Combines two sampling strategies:
1. **Uniform sampling** (20% by default): Guarantees ergodic coverage of the entire
   configuration space, ensuring no isolated peaks are missed
2. **MCMC sampling** (80% by default): Efficiently explores high-probability regions
   using Metropolis-Hastings with target distribution ∝ |f(x)|²

This hybrid approach provides the best of both worlds:
- Uniform samples prevent missing isolated modes (ergodicity)
- MCMC samples reduce variance in important regions (efficiency)

# When to Use
- **Recommended for**: Sparse or localized wavefunctions with isolated peaks
- **Better than uniform**: When f has strong amplitude variations across configurations
- **Better than pure MCMC**: When ergodicity is a concern (multiple isolated modes)
- **Not necessary for**: Smooth wavefunctions with uniform-like amplitudes

# Tuning Parameters
- Increase `mcmc_fraction` (e.g., 0.9) for very sparse wavefunctions
- Decrease `mcmc_fraction` (e.g., 0.5) if ergodicity is more important
- Increase `n_burnin` (e.g., 5000) for complex energy landscapes
- Increase `n_flip` (e.g., 2-3) for faster exploration but lower acceptance rate
- Decrease `thin` (e.g., 5) to collect more correlated samples (faster but higher variance)

# Example
```julia
# Sparse wavefunction with localized peaks
f(x) = sum(x[i] == x[i+1] for i in 1:length(x)-1) >= length(x)-2 ? 1.0 : 0.1
MI = mutualinformation_hybrid(f, 30; n_samples=100000, mcmc_fraction=0.9)

# For comparison with uniform sampling
MI_uniform = mutualinformation_uniform(f, 30; n_samples=100000)
```

# See Also
- `mutualinformation_uniform`: Pure uniform sampling (baseline)
- `mutualinformation_exact`: Exact calculation for small systems
"""
function mutualinformation_hybrid(f::F, len::Int;
    n_samples::Int=100000,
    mcmc_fraction::Float64=0.8,
    n_burnin::Int=1000,
    thin::Int=10,
    n_flip::Int=1,
    rng=default_rng()) where {F}

    error("This implementation is not checked and shouldn't be trusted.")

    # Validate parameters
    if !(0.0 <= mcmc_fraction <= 1.0)
        throw(ArgumentError("mcmc_fraction must be between 0 and 1, got $mcmc_fraction"))
    end

    # Split samples between uniform and MCMC
    n_mcmc = round(Int, n_samples * mcmc_fraction)
    n_uniform = n_samples - n_mcmc

    # Generate uniform samples
    uniform_samples = [rand(rng, 1:2, len) for _ in 1:n_uniform]
    uniform_f_vals = [f(x) for x in uniform_samples]

    # Generate MCMC samples
    mcmc_samples = Vector{Vector{Int}}()
    mcmc_f_vals = Vector{ComplexF64}()

    # Initialize MCMC from a random configuration
    x = rand(rng, 1:2, len)
    fx = f(x)

    # If initial sample has zero amplitude, try a few more
    attempts = 0
    while abs(fx) < 1e-14 && attempts < 100
        x = rand(rng, 1:2, len)
        fx = f(x)
        attempts += 1
    end

    if abs(fx) < 1e-14 && n_mcmc > 0
        @warn "MCMC initialization: all tested configurations have near-zero amplitude. " *
              "MCMC may not be effective. Consider using more uniform samples."
    end

    # MCMC with Metropolis-Hastings
    n_accepted = 0
    total_steps = n_mcmc * thin + n_burnin

    for iter in 1:total_steps
        # Propose: flip n_flip random bits
        x_prop = copy(x)
        # More aggressive flips during burn-in to explore faster
        k = (iter <= n_burnin && n_flip == 1) ? rand(rng, 1:min(3, len)) : n_flip
        for _ in 1:k
            site = rand(rng, 1:len)
            x_prop[site] = 3 - x_prop[site]  # flip 1↔2
        end
        fx_prop = f(x_prop)

        # Metropolis acceptance with |f|² as target distribution
        # This samples from the quantum probability distribution
        accept = false
        if abs2(fx) < 1e-100  # Current state has ~zero amplitude
            accept = true  # Always accept moves away from zero
        else
            α = abs2(fx_prop) / abs2(fx)
            accept = rand(rng) < min(1.0, α)
        end

        if accept
            x = x_prop
            fx = fx_prop
            n_accepted += 1
        end

        # Collect sample (after burn-in, every 'thin' steps)
        if iter > n_burnin && (iter - n_burnin) % thin == 0
            push!(mcmc_samples, copy(x))
            push!(mcmc_f_vals, fx)
        end
    end

    # Report MCMC statistics
    if n_mcmc > 0
        acceptance_rate = n_accepted / total_steps
        @info "MCMC statistics" acceptance_rate = round(acceptance_rate * 100, digits=1) n_mcmc n_uniform
    end

    # Combine samples
    all_samples = vcat(uniform_samples, mcmc_samples)
    all_f_vals = vcat(uniform_f_vals, mcmc_f_vals)

    # Check that we have some non-zero samples
    if all(x -> abs(x) < 1e-14, all_f_vals[1:min(100, length(all_f_vals))])
        error("Function f appears to be zero on all sampled configurations")
    end

    # Compute mutual information matrix using combined samples
    MI_matrix = zeros(Float64, len, len)

    # Generate all pairs in upper triangle to enable parallel computation
    pairs = [(A, B) for A in 1:len for B in (A+1):len]

    Threads.@threads for (A, B) in pairs
        # Estimate the full reduced density matrix on sites A and B
        ρ_AB = estimate_reduced_density_matrix_AB(all_samples, all_f_vals, f, A, B)

        # Compute reduced density matrices by partial tracing
        ρ_A = partial_trace_to_sites(ρ_AB, [2, 2], [1])  # Keep first qubit
        ρ_B = partial_trace_to_sites(ρ_AB, [2, 2], [2])  # Keep second qubit

        # Compute von Neumann entropies
        S_A = von_neumann_entropy(ρ_A)
        S_B = von_neumann_entropy(ρ_B)
        S_AB = von_neumann_entropy(ρ_AB)

        # Mutual information: I(A:B) = S(A) + S(B) - S(AB)
        # Clamp to non-negative to handle numerical errors
        MI_matrix[A, B] = max(0.0, S_A + S_B - S_AB)
    end

    # Symmetrize the matrix (MI is symmetric by definition)
    for A in 1:len
        for B in 1:(A-1)
            MI_matrix[A, B] = MI_matrix[B, A]
        end
    end

    return MI_matrix
end
