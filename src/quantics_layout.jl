# Quantics bit-layout optimization from a mutual information matrix.
#
# A quantics tensor-train (MPS) representation of a function places its R bits on a
# 1D chain. The bond dimension χ at a cut is bounded by the entanglement across that
# cut, and the cut entanglement is bounded by the mutual information across it
# (Schuch-Wolf-Verstraete-Cirac: I(left:right) ≤ 2 log χ). To keep bond dimensions
# small we therefore want an ordering that places highly mutually-informative bits
# close together. With weights W[i,j] = I(site_i : site_j), the cost
#   ∑_{i<j} W[π(i), π(j)] · |π(i) - π(j)|
# is the Minimum Linear Arrangement objective, and a degree-2 spanning tree is a
# Hamiltonian path = a linear arrangement. `solve_oct(W; max_deg=2)` now delegates
# to the canonical MinLA solver in `OptimalCommunicationTree`.
#
# This module provides the pieces that connect an MI matrix to a concrete bit
# ordering and that validate an ordering against *actual* bond dimensions:
#   - amplitude_tensor(f, localdims):  build the exact state tensor
#   - bond_dimensions(ψ, perm):       exact MPS bond dims (ranks of unfoldings)
#   - cut_entropies(ψ, perm):         entanglement entropies across each cut
#   - mi_ordering(W; max_deg=2,...):  Hamiltonian path from the MI matrix
#   - minla_cost(W, perm):            the linear-arrangement objective
#
# All of these are pure linear algebra / graph ops; no QuanticsGrids or TCI
# dependency, so they live in the core package.

using LinearAlgebra: svdvals, norm

"""
    amplitude_tensor(f, localdims::AbstractVector{<:Integer}) -> Array

Build the exact state tensor `ψ[b₁, ..., b_L] = f([b₁, ..., b_L])` with
`b_k ∈ {1, ..., localdims[k]}`. The axis order matches the site order used by
`mutualinformation` (site `k` ↔ `localdims[k]` ↔ axis `k`).
"""
function amplitude_tensor(f, localdims::AbstractVector{<:Integer})
    L = length(localdims)
    localdims = collect(localdims)
    total = prod(localdims)
    vals = Vector{Any}(undef, total)
    @inbounds for i in 1:total
        vals[i] = f(index_to_config(i, localdims))
    end
    T = promote_type(Float64, (typeof(v) for v in vals)...)
    ψ = Array{T,L}(undef, Tuple(localdims))
    @inbounds for i in 1:total
        ψ[i] = vals[i]
    end
    return ψ
end

"""
    bond_dimensions(ψ::AbstractArray{T,L}, perm=1:L; rtol=1e-12) -> Vector{Int}

Exact MPS bond dimensions for the site ordering `perm`: at cut `k` (between sites
`k` and `k+1` of the chain) this is the rank of the unfolding
`reshape(ψ[:, perm], prod(dims[1:k]), prod(dims[k+1:L]))`, i.e. the number of
singular values exceeding `rtol * σ_max`.

These are the *minimal* bond dimensions of an exact MPS for `ψ` under ordering
`perm` — the ground-truth figure of merit for layout optimization on small systems.
"""
function bond_dimensions(ψ::AbstractArray{T,L}, perm=1:L; rtol::Real=1e-12) where {T,L}
    ψp = permutedims(ψ, perm)
    dims = size(ψp)
    χ = zeros(Int, L - 1)
    for k in 1:(L - 1)
        m = prod(dims[1:k]); n = prod(dims[k+1:L])
        s = svdvals(reshape(ψp, m, n))
        isempty(s) && continue
        thr = rtol * s[1]
        χ[k] = count(>(thr), s)
    end
    return χ
end

"""
    cut_entropies(ψ::AbstractArray{T,L}, perm=1:L) -> Vector{Float64}

Von Neumann entanglement entropies (nats) across each MPS cut for the site ordering
`perm`, computed from the singular-value spectrum of the unfoldings of the
(renormalized) state. The effective bond dimension needed to capture the cut to
truncation ε scales like `exp(S_k)` up to log-corrections, so `max S_k` and
`sum S_k` are proxies for worst-case and total bond dimension.
"""
function cut_entropies(ψ::AbstractArray{T,L}, perm=1:L) where {T,L}
    ψp = permutedims(ψ, perm)
    dims = size(ψp)
    nrm = norm(ψp)
    S = zeros(Float64, L - 1)
    for k in 1:(L - 1)
        m = prod(dims[1:k]); n = prod(dims[k+1:L])
        s = svdvals(reshape(ψp, m, n)) ./ nrm
        p = s .^ 2
        p = p[p .> 1e-14]
        S[k] = -sum(p .* log.(p))
    end
    return S
end

"""
    path_to_ordering(tree) -> Vector{Int}

Backward-compatible alias for `ordering_from_path(tree)`.
"""
path_to_ordering(tree) = ordering_from_path(tree)

"""
    mi_ordering(W; max_deg=2, verbose=false, kwargs...) -> (perm, cost, tree)

Find a site ordering that keeps mutually-informative sites close together by
solving weighted MinLA on the MI matrix. `max_deg` must be 2 for nontrivial
layouts; this is the Hamiltonian-path/MPS-chain case.
"""
function mi_ordering(W::AbstractMatrix{<:Real}; max_deg::Int=2, verbose::Bool=false, kwargs...)
    tree, cost = solve_oct(W; max_deg, verbose, kwargs...)
    return ordering_from_path(tree), cost, tree
end
