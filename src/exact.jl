# Exact mutual information calculation for small systems
# Uses full density matrix construction and partial tracing

"""
    mutualinformation_exact(::Type{T}, f::F, localdims::AbstractVector{<:Integer}) where {T,F}

Exact calculation of mutual information for all pairs of sites.

`f` must be a function and take exactly one argument, a `Vector{Int}`. `f(x)`
with `length(x) == len`, `x[i] ∈ (1, 2)` for any `i ∈ eachindex(x)` should
return a value of type `T`.

The function will form the full density matrix `ρ` (with `size(ρ) == (2^L, 2^L)`)
defined by `ρ[i, j] == f(x_values[i]) * conj(f(x_values[j]))` where
`x_values::Vector{Vector{Int}}` is a lexicographically arranged collection of all
possible function inputs (thus `length(x_values) == 2^L`). It will then for each
combination of integers `A` and `B` (both of value at least `1` and at most `L`)
form the partial trace of the full density matrix over all bits excluding `A` and
`B`, dubbed `ρ_AB`. This is then further reduced to the one-bit density matrices
`ρ_A` (tracing out `B`) and `ρ_B` (tracing out `A`).
Finally, the mutual information between `A` and `B` is computed by
`-tr(ρ_A * log(ρ_A)) - tr(ρ_B * log(ρ_B)) + tr(ρ_AB * log(ρ_AB))`.
The mutual informations of all such `A` - `B` pairs is then arranged in an
`len` x `len` matrix and returned.

# Complexity
- Time: O(L³ × d^(2L)) where L is length and d is local dimension
- Space: O(d^(2L))

# Recommended Usage
- Systems with length < 15
- When exact values are required
- High precision needed

For larger systems (L ≥ 15), use `mutualinformation_sampled` instead.

This function is typically not called directly - use `mutualinformation()` which
automatically selects the appropriate method based on system size.
"""
function mutualinformation_exact(::Type{T}, f::F, localdims::AbstractVector{<:Integer}) where {T,F}
    L = length(localdims)
    total_dim = prod(localdims)

    # Generate all possible configurations in lexicographic order
    x_values = [index_to_config(i, localdims) for i in 0:(total_dim-1)]

    # Build full density matrix: ρ[i,j] = f(x_values[i]) * conj(f(x_values[j]))
    ρ = [f(x_values[i]) * conj(f(x_values[j])) for i in 1:total_dim, j in 1:total_dim]

    # Normalize density matrix (ensure trace = 1)
    ρ = ρ / tr(ρ)

    # Compute mutual information for all pairs of sites
    MI_matrix = zeros(Float64, L, L)

    for A in 1:L
        for B in 1:L
            if A == B
                MI_matrix[A, B] = 0.0
            else
                # Compute reduced density matrices
                ρ_AB = partial_trace_to_sites(ρ, localdims, [A, B])
                ρ_A = partial_trace_second(ρ_AB, localdims[A], localdims[B])
                ρ_B = partial_trace_first(ρ_AB, localdims[A], localdims[B])

                # Compute von Neumann entropies
                S_A = von_neumann_entropy(ρ_A)
                S_B = von_neumann_entropy(ρ_B)
                S_AB = von_neumann_entropy(ρ_AB)

                # Mutual information: I(A:B) = S(A) + S(B) - S(AB)
                MI_matrix[A, B] = S_A + S_B - S_AB
            end
        end
    end

    return MI_matrix
end

"""
    index_to_config(idx::Int, localdims::AbstractVector{<:Integer}) -> Vector{Int}

Convert a linear index to a configuration vector where config[j] ∈ [1, localdims[j]].
"""
function index_to_config(idx::Int, localdims::AbstractVector{<:Integer})
    L = length(localdims)
    config = Vector{Int}(undef, L)
    for j in 1:L
        config[j] = (idx % localdims[j]) + 1
        idx = div(idx, localdims[j])
    end
    return config
end

"""
    config_to_index(config::Vector{Int}, localdims::AbstractVector{<:Integer}) -> Int

Convert a configuration vector to a linear index.
"""
function config_to_index(config::Vector{Int}, localdims::AbstractVector{<:Integer})
    idx = 0
    stride = 1
    for j in 1:length(config)
        idx += (config[j] - 1) * stride
        stride *= localdims[j]
    end
    return idx
end

"""
    partial_trace_to_sites(ρ, localdims, sites) -> Matrix

Trace out all sites except those specified in `sites`.
Returns the reduced density matrix on the specified sites.
"""
function partial_trace_to_sites(ρ, localdims::AbstractVector{<:Integer}, sites::Vector{Int})
    L = length(localdims)
    total_dim = size(ρ, 1)

    # Dimensions of kept and traced sites
    kept_dims = [localdims[s] for s in sites]
    reduced_dim = prod(kept_dims)

    ρ_reduced = zeros(eltype(ρ), reduced_dim, reduced_dim)

    # Iterate over all pairs of full configurations
    for i in 0:(total_dim-1)
        config_i = index_to_config(i, localdims)

        for j in 0:(total_dim-1)
            config_j = index_to_config(j, localdims)

            # Check if traced-out sites are the same in both configs
            traced_match = true
            for k in 1:L
                if !(k in sites) && config_i[k] != config_j[k]
                    traced_match = false
                    break
                end
            end

            if traced_match
                # Extract configurations on kept sites
                config_i_reduced = [config_i[s] for s in sites]
                config_j_reduced = [config_j[s] for s in sites]

                # Convert to reduced indices
                idx_i_reduced = config_to_index(config_i_reduced, kept_dims)
                idx_j_reduced = config_to_index(config_j_reduced, kept_dims)

                ρ_reduced[idx_i_reduced + 1, idx_j_reduced + 1] += ρ[i + 1, j + 1]
            end
        end
    end

    return ρ_reduced
end

"""
    partial_trace_first(ρ_AB, dim_A, dim_B) -> Matrix

Trace out the first subsystem from a two-site density matrix.
Returns ρ_B.
"""
function partial_trace_first(ρ_AB, dim_A::Int, dim_B::Int)
    ρ_B = zeros(eltype(ρ_AB), dim_B, dim_B)
    for a in 1:dim_A
        for b1 in 1:dim_B
            for b2 in 1:dim_B
                idx1 = (a-1)*dim_B + b1
                idx2 = (a-1)*dim_B + b2
                ρ_B[b1, b2] += ρ_AB[idx1, idx2]
            end
        end
    end
    return ρ_B
end

"""
    partial_trace_second(ρ_AB, dim_A, dim_B) -> Matrix

Trace out the second subsystem from a two-site density matrix.
Returns ρ_A.
"""
function partial_trace_second(ρ_AB, dim_A::Int, dim_B::Int)
    ρ_A = zeros(eltype(ρ_AB), dim_A, dim_A)
    for b in 1:dim_B
        for a1 in 1:dim_A
            for a2 in 1:dim_A
                idx1 = (a1-1)*dim_B + b
                idx2 = (a2-1)*dim_B + b
                ρ_A[a1, a2] += ρ_AB[idx1, idx2]
            end
        end
    end
    return ρ_A
end

"""
    von_neumann_entropy(ρ) -> Float64

Compute the von Neumann entropy S(ρ) = -tr(ρ log ρ) of a density matrix.
"""
function von_neumann_entropy(ρ)
    # Compute eigenvalues, treating ρ as Hermitian for numerical stability
    eigvals_ρ = eigvals(Hermitian(ρ))
    # Filter out numerical zeros to avoid log(0)
    eigvals_ρ = filter(λ -> λ > 1e-14, eigvals_ρ)
    # Compute entropy: S = -sum(λ log λ)
    return -sum(λ * log(λ) for λ in eigvals_ρ)
end
