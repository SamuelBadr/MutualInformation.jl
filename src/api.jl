# Unified API for mutual information calculation
# Automatically selects between exact and sampling methods

"""
    mutualinformation(f::F, len::Int;
                     method=:auto,
                     threshold=14,
                     n_samples=100000) where {F}

Calculate mutual information between all pairs of sites in a quantum state.

Automatically selects the appropriate method (exact or sampling) based on system size.

# Arguments
- `f`: Function taking `Vector{Int}` (values in {1,2,...}) and returning a numeric value
- `len`: Number of sites/bits in the system

# Keyword Arguments
- `method::Symbol`: Method to use (`:auto`, `:exact`, `:uniform`, `:hybrid`, or `:sampled`)
  - `:auto` (default): Automatically choose based on `len` and `threshold`
  - `:exact`: Force exact calculation (may be infeasible for large `len`)
  - `:uniform`: Force uniform sampling-based approximation
  - `:hybrid`: Force hybrid MCMC+uniform sampling (best for sparse wavefunctions)
  - `:sampled`: Deprecated alias for `:uniform`
- `threshold::Int`: System size threshold for auto method selection (default: 10)
  - Systems with `len ≤ threshold` use exact method
  - Systems with `len > threshold` use uniform sampling method
- `n_samples::Int`: Number of samples for sampling methods (default: 100000)
  - Only used when method is `:uniform`, `:hybrid`, `:sampled`, or auto-selected
- `mcmc_fraction::Float64`: For `:hybrid` method, fraction of MCMC samples (default: 0.8)
- Additional MCMC parameters: `n_burnin`, `thin`, `n_flip` (see `mutualinformation_hybrid`)

# Returns
- `len × len` matrix of mutual information values in nats

# Method Selection
The function automatically chooses the best method based on system size:
- **Exact method** (`len ≤ 10`):
  - Constructs full density matrix
  - Exact values to machine precision
  - Complexity: O(L³ × 4^L)
  - Suitable for small systems

- **Uniform sampling** (`len > 10`):
  - Monte Carlo sampling with uniform configuration sampling
  - Typical accuracy: 10-20% relative error
  - Complexity: O(L² × n_samples)
  - Required for large systems
  - Guaranteed ergodic coverage

- **Hybrid sampling** (manual selection):
  - Combines uniform (20%) and MCMC (80%) sampling
  - Better for sparse/localized wavefunctions
  - Same complexity but lower variance for peaked distributions
  - Maintains ergodicity while improving efficiency

# Examples
```julia
# Small system - automatically uses exact method
f(x) = exp(-sum((x[i] - 1.5)^2 for i in eachindex(x)))
MI = mutualinformation(f, 10)

# Large system - automatically uses uniform sampling
MI = mutualinformation(f, 30)

# Force exact method for medium system
MI = mutualinformation(f, 15; method=:exact)

# Force uniform sampling with more samples
MI = mutualinformation(f, 10; method=:uniform, n_samples=200000)

# Use hybrid sampling for sparse wavefunction
f_sparse(x) = sum(x[i] == x[i+1] for i in 1:length(x)-1) >= length(x)-2 ? 1.0 : 0.1
MI = mutualinformation(f_sparse, 30; method=:hybrid, mcmc_fraction=0.9)

# Adjust auto-selection threshold
MI = mutualinformation(f, 16; threshold=16)  # Uses exact for len=16
```

# See Also
- `mutualinformation_exact`: Direct access to exact method
- `mutualinformation_uniform`: Direct access to uniform sampling method
- `mutualinformation_hybrid`: Direct access to hybrid MCMC+uniform method
"""
function mutualinformation(f::F, len::Int;
    method::Symbol=:auto,
    threshold::Int=10,
    n_samples::Int=100_000,
    mcmc_fraction::Float64=0.8,
    n_burnin::Int=1000,
    thin::Int=10,
    n_flip::Int=1,
    rng=default_rng()) where {F}

    # Validate method parameter
    if !(method in (:auto, :exact, :uniform, :hybrid, :sampled))
        throw(ArgumentError("method must be :auto, :exact, :uniform, :hybrid, or :sampled, got :$method"))
    end

    # Handle deprecated :sampled alias
    if method == :sampled
        method = :uniform
    end

    # Determine which method to use
    if method == :auto
        use_method = len <= threshold ? :exact : :uniform
    else
        use_method = method
    end

    # Warn if using exact for large systems
    if use_method == :exact && len > 20
        @warn "Using exact method for len=$len may be very slow and memory-intensive. " *
              "Consider using method=:uniform or method=:hybrid."
    end

    # Dispatch to appropriate method
    if use_method == :exact
        # For exact method, we need localdims vector
        localdims = fill(2, len)
        return mutualinformation_exact(f, localdims)
    elseif use_method == :uniform
        return mutualinformation_uniform(f, len; n_samples, rng)
    else  # method == :hybrid
        return mutualinformation_hybrid(f, len; n_samples, mcmc_fraction, n_burnin, thin, n_flip, rng)
    end
end

"""
    mutualinformation(f::F, localdims::AbstractVector{<:Integer};
                     method=:auto,
                     threshold=14,
                     n_samples=100000) where {F}

Calculate mutual information with non-uniform local dimensions.

This variant allows specifying different local dimensions for each site (e.g.,
mixing qubits and qutrits). Currently only supports exact method.

# Arguments
- `f`: Function taking `Vector{Int}` where `x[i] ∈ {1, ..., localdims[i]}`
- `localdims`: Vector of local dimensions for each site

# Keyword Arguments
- `method::Symbol`: Method to use (default: :auto)
- `threshold::Int`: Size threshold for auto selection (default: 10)
- `n_samples::Int`: Number of samples (ignored for non-uniform dimensions)

# Note
Non-uniform local dimensions currently only support exact method. The sampling
methods (uniform and hybrid) are designed for uniform binary systems.

# Examples
```julia
# Mixed qubit-qutrit system
f(x) = exp(-sum((x[i] - 1.5)^2 for i in eachindex(x)))
MI = mutualinformation(f, [2, 2, 3, 2])  # 3 qubits + 1 qutrit
```
"""
function mutualinformation(f::F, localdims::AbstractVector{<:Integer};
    method::Symbol=:auto,
    threshold::Int=10,
    n_samples::Int=100000,
    mcmc_fraction::Float64=0.8,
    n_burnin::Int=1000,
    thin::Int=10,
    n_flip::Int=1,
    rng=default_rng()) where {F}

    L = length(localdims)

    # Check if all dimensions are 2 (uniform binary)
    all_binary = all(d == 2 for d in localdims)

    # Validate method parameter
    if !(method in (:auto, :exact, :uniform, :hybrid, :sampled))
        throw(ArgumentError("method must be :auto, :exact, :uniform, :hybrid, or :sampled, got :$method"))
    end

    # Handle deprecated :sampled alias
    if method == :sampled
        method = :uniform
    end

    # Non-uniform dimensions only support exact method
    if !all_binary && method in (:uniform, :hybrid)
        throw(ArgumentError("Sampling methods only support uniform binary systems (all localdims=2). " *
                            "Use method=:exact for non-uniform dimensions."))
    end

    # Determine which method to use
    if method == :auto
        use_method = (!all_binary || L <= threshold) ? :exact : :uniform
    else
        use_method = method
    end

    # Warn if using exact for large systems
    if use_method == :exact && L > 20
        @warn "Using exact method for len=$L may be very slow and memory-intensive. " *
              "Consider reducing system size or using method=:uniform."
    end

    # Dispatch to appropriate method
    if use_method == :exact
        return mutualinformation_exact(f, localdims)
    elseif use_method == :uniform
        # All binary, use uniform sampling
        return mutualinformation_uniform(f, L; n_samples, rng)
    else  # method == :hybrid
        return mutualinformation_hybrid(f, L; n_samples, mcmc_fraction, n_burnin, thin, n_flip, rng)
    end
end
