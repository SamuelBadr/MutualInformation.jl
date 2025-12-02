# Shared utility functions for mutual information calculations

"""
    von_neumann_entropy(ρ) -> Float64

Compute the von Neumann entropy S(ρ) = -tr(ρ log ρ) of a density matrix.

The entropy is calculated from the eigenvalues: S = -∑ λᵢ log λᵢ,
where the sum is over all positive eigenvalues.

# Arguments
- `ρ`: Density matrix (treated as Hermitian for numerical stability)

# Returns
- Von Neumann entropy in nats

# Notes
- Eigenvalues below 1e-14 are filtered out to avoid log(0)
- Natural logarithm is used (entropy in nats, not bits)
"""
function von_neumann_entropy(ρ)
    # Compute eigenvalues, treating ρ as Hermitian for numerical stability
    eigvals_ρ = eigvals!(Hermitian(ρ))
    # Filter out numerical zeros to avoid log(0)
    eigvals_ρ = filter(λ -> λ > 1e-14, eigvals_ρ)
    # Compute entropy: S = -sum(λ log λ)
    return -sum(λ * log(λ) for λ in eigvals_ρ)
end

"""
    index_to_config(idx::Int, localdims::AbstractVector{<:Integer}) -> Vector{Int}

Convert a 1-based linear index to a configuration vector.

Uses mixed-radix representation: config[j] ∈ {1, ..., localdims[j]}.
"""
function index_to_config(idx::Int, localdims::AbstractVector{<:Integer})
    L = length(localdims)
    config = Vector{Int}(undef, L)
    idx = idx - 1  # Convert to 0-based for calculation
    for j in 1:L
        config[j] = (idx % localdims[j]) + 1
        idx = div(idx, localdims[j])
    end
    return config
end

"""
    config_to_index(config::Vector{Int}, localdims::AbstractVector{<:Integer}) -> Int

Convert a configuration vector to a 1-based linear index.
"""
function config_to_index(config::Vector{Int}, localdims::AbstractVector{<:Integer})
    idx = 0
    stride = 1
    for j in 1:length(config)
        idx += (config[j] - 1) * stride
        stride *= localdims[j]
    end
    return idx + 1  # Convert to 1-based index
end

"""
    partial_trace_to_sites(ρ, localdims, keptsites) -> Matrix

Compute partial trace to obtain reduced density matrix on specified sites.

# Arguments
- `ρ`: Full density matrix
- `localdims`: Vector of local dimensions for each site
- `keptsites`: Indices of sites to keep (all others traced out)

# Returns
- Reduced density matrix on the kept sites
"""
function partial_trace_to_sites(ρ, localdims::AbstractVector{<:Integer}, keptsites::Vector{Int})
    L = length(localdims)
    total_dim = size(ρ, 1)

    # Dimensions of kept and traced sites
    kept_dims = [localdims[s] for s in keptsites]
    reduced_dim = prod(kept_dims)
    allsites = collect(1:L)
    tracedoutsites = setdiff(allsites, keptsites)

    ρ_reduced = zeros(eltype(ρ), reduced_dim, reduced_dim)

    # Iterate over all pairs of full configurations
    for i in 1:total_dim
        config_i = index_to_config(i, localdims)

        for j in 1:total_dim
            config_j = index_to_config(j, localdims)

            # Check if traced-out sites are the same in both configs
            traced_match = all(tracedoutsites) do k
                config_i[k] == config_j[k]
            end
            traced_match || continue

            # Extract configurations on keptsites
            config_i_reduced = [config_i[s] for s in keptsites]
            config_j_reduced = [config_j[s] for s in keptsites]

            # Convert to reduced indices
            idx_i_reduced = config_to_index(config_i_reduced, kept_dims)
            idx_j_reduced = config_to_index(config_j_reduced, kept_dims)

            ρ_reduced[idx_i_reduced, idx_j_reduced] += ρ[i, j]
        end
    end

    return ρ_reduced
end
