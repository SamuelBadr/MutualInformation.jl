# Exact mutual information calculation for small systems
# Uses full density matrix construction and partial tracing

"""
    mutualinformation_exact(f::F, localdims::AbstractVector{<:Integer}) where {F}

Exact calculation of mutual information for all pairs of sites using full
density matrix construction.

# Arguments
- `f`: Wavefunction amplitude function taking `Vector{Int}` where
       `x[i] ∈ {1, ..., localdims[i]}` and returning a numeric value (real or complex)
- `localdims`: Vector of local dimensions for each site (e.g., [2,2,2] for qubits)

# Returns
- `L × L` symmetric matrix of mutual information values in nats

# Method
1. Construct full density matrix: ρ[i,j] = f(xᵢ) × conj(f(xⱼ)) for all configurations
2. For each pair (A, B), compute reduced density matrices ρ_AB, ρ_A, ρ_B via partial trace
3. Compute von Neumann entropies S(A), S(B), S(AB) from eigenvalues
4. Return mutual information: I(A:B) = S(A) + S(B) - S(AB)

# Complexity
- Time: O(L³ × d^(2L)) where L is length and d is local dimension
- Space: O(d^(2L))

# Recommended Usage
- Systems with length < 15
- When exact values are required (machine precision)
- Non-uniform local dimensions

For larger systems (L ≥ 15), use `mutualinformation_sampled` instead.

**Note:** Typically not called directly - use `mutualinformation()` which
automatically selects the appropriate method based on system size.
"""
function mutualinformation_exact(f::F, localdims::AbstractVector{<:Integer}) where {F}
    L = length(localdims)
    total_dim = prod(localdims)

    # Generate all possible configurations in lexicographic order
    x_values = [index_to_config(i, localdims) for i in 1:total_dim]

    # Build full density matrix: ρ[i,j] = f(x_values[i]) * conj(f(x_values[j]))
    ψ = f.(x_values)
    ρ = ψ * ψ'

    # Normalize density matrix (ensure trace = 1)
    ρ = ρ / tr(ρ)

    # Compute mutual information for all pairs of sites
    MI_matrix = zeros(Float64, L, L)

    # Generate all pairs to enable parallel computation
    pairs = [(A, B) for A in 1:L for B in 1:L]

    Threads.@threads for (A, B) in pairs
        if A == B
            MI_matrix[A, B] = 0.0
        else
            # Compute reduced density matrices
            ρ_AB = partial_trace_to_sites(ρ, localdims, [A, B])
            ρ_A = partial_trace_to_sites(ρ_AB, [localdims[A], localdims[B]], [1])
            ρ_B = partial_trace_to_sites(ρ_AB, [localdims[A], localdims[B]], [2])

            # Compute von Neumann entropies
            S_A = von_neumann_entropy(ρ_A)
            S_B = von_neumann_entropy(ρ_B)
            S_AB = von_neumann_entropy(ρ_AB)

            # Mutual information: I(A:B) = S(A) + S(B) - S(AB)
            # Clamp to non-negative to handle numerical errors
            MI_matrix[A, B] = max(0.0, S_A + S_B - S_AB)
        end
    end

    return MI_matrix
end


