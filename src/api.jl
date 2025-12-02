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
- `method::Symbol`: Method to use (`:auto`, `:exact`, or `:sampled`)
  - `:auto` (default): Automatically choose based on `len` and `threshold`
  - `:exact`: Force exact calculation (may be infeasible for large `len`)
  - `:sampled`: Force sampling-based approximation
- `threshold::Int`: System size threshold for auto method selection (default: 14)
  - Systems with `len ≤ threshold` use exact method
  - Systems with `len > threshold` use sampling method
- `n_samples::Int`: Number of samples for sampling method (default: 100000)
  - Only used when method is `:sampled` or auto-selected

# Returns
- `len × len` matrix of mutual information values in nats

# Method Selection
The function automatically chooses the best method based on system size:
- **Exact method** (`len ≤ 14`):
  - Constructs full density matrix
  - Exact values to machine precision
  - Complexity: O(L³ × 4^L)
  - Suitable for small systems

- **Sampling method** (`len > 14`):
  - Monte Carlo sampling approximation
  - Typical accuracy: 10-20% relative error
  - Complexity: O(L² × n_samples)
  - Required for large systems

# Examples
```julia
# Small system - automatically uses exact method
f(x) = exp(-sum((x[i] - 1.5)^2 for i in eachindex(x)))
MI = mutualinformation(f, 10)

# Large system - automatically uses sampling method
MI = mutualinformation(f, 30)

# Force exact method for medium system
MI = mutualinformation(f, 15; method=:exact)

# Force sampling with more samples
MI = mutualinformation(f, 10; method=:sampled, n_samples=200000)

# Adjust auto-selection threshold
MI = mutualinformation(f, 16; threshold=16)  # Uses exact for len=16
```

# See Also
- `mutualinformation_exact`: Direct access to exact method
- `mutualinformation_sampled`: Direct access to sampling method
"""
function mutualinformation(f::F, len::Int;
    method::Symbol=:auto,
    threshold::Int=14,
    n_samples::Int=100000) where {F}

    # Validate method parameter
    if !(method in (:auto, :exact, :sampled))
        throw(ArgumentError("method must be :auto, :exact, or :sampled, got :$method"))
    end

    # Determine which method to use
    use_exact = if method == :auto
        len <= threshold
    elseif method == :exact
        # Warn if using exact for large systems
        if len > 20
            @warn "Using exact method for len=$len may be very slow and memory-intensive. " *
                  "Consider using method=:sampled or increasing the threshold."
        end
        true
    else  # method == :sampled
        false
    end

    # Dispatch to appropriate method
    if use_exact
        # For exact method, we need localdims vector
        localdims = fill(2, len)
        return mutualinformation_exact(f, localdims)
    else
        return mutualinformation_sampled(f, len; n_samples=n_samples)
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
- `threshold::Int`: Size threshold for auto selection (default: 14)
- `n_samples::Int`: Number of samples (ignored for non-uniform dimensions)

# Note
Non-uniform local dimensions currently only support exact method. The sampling
method is designed for uniform binary systems.

# Examples
```julia
# Mixed qubit-qutrit system
f(x) = exp(-sum((x[i] - 1.5)^2 for i in eachindex(x)))
MI = mutualinformation(f, [2, 2, 3, 2])  # 3 qubits + 1 qutrit
```
"""
function mutualinformation(f::F, localdims::AbstractVector{<:Integer};
    method::Symbol=:auto,
    threshold::Int=14,
    n_samples::Int=100000) where {F}

    L = length(localdims)

    # Check if all dimensions are 2 (uniform binary)
    all_binary = all(d == 2 for d in localdims)

    # Validate method parameter
    if !(method in (:auto, :exact, :sampled))
        throw(ArgumentError("method must be :auto, :exact, or :sampled, got :$method"))
    end

    # Non-uniform dimensions only support exact method
    if !all_binary && method == :sampled
        throw(ArgumentError("Sampling method only supports uniform binary systems (all localdims=2). " *
                            "Use method=:exact for non-uniform dimensions."))
    end

    # Determine which method to use
    use_exact = if method == :auto
        !all_binary || L <= threshold
    elseif method == :exact
        if L > 20
            @warn "Using exact method for len=$L may be very slow and memory-intensive. " *
                  "Consider reducing system size."
        end
        true
    else  # method == :sampled
        false
    end

    # Dispatch to appropriate method
    if use_exact
        return mutualinformation_exact(f, localdims)
    else
        # All binary, use sampling
        return mutualinformation_sampled(f, L; n_samples=n_samples)
    end
end
