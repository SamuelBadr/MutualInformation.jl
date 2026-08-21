# Generalized mutual-information diagnostics for exact quantics tensors.
#
# Pairwise bit MI is only a graph surrogate. For many-body constraints we need
# set/block quantities:
#   S(A)                       entropy of an arbitrary subset of sites
#   I(A:B)                     block mutual information
#   I(A:B|C)                   conditional mutual information
#   I(A:B∪C)-I(A:B)-I(A:C)     simple synergy / non-additivity diagnostic
#
# These routines operate on an exact amplitude tensor ψ and use SVD unfoldings,
# so they are intended for small/coarse resolutions and calibration.

"""
    subsystem_entropy(ψ, sites) -> Float64

Von Neumann entropy `S(sites)` of a pure-state amplitude tensor `ψ`, computed as
the entropy of the singular values of the unfolding `sites | complement(sites)`.
The tensor is normalized internally. `sites` may be empty or all sites, in which
case the entropy is zero.
"""
function subsystem_entropy(ψ::AbstractArray{T,L}, sites::AbstractVector{<:Integer}) where {T,L}
    A = sort(collect(Int, sites))
    all(1 .<= A .<= L) || throw(ArgumentError("sites must lie in 1:$L"))
    length(unique(A)) == length(A) || throw(ArgumentError("sites contains duplicates"))
    (isempty(A) || length(A) == L) && return 0.0
    B = setdiff(collect(1:L), A)
    perm = [A; B]
    ψp = permutedims(ψ, perm)
    dims = size(ψp)
    m = prod(dims[1:length(A)])
    n = prod(dims[(length(A)+1):end])
    nrm = norm(ψp)
    iszero(nrm) && throw(ArgumentError("cannot compute entropy of zero tensor"))
    s = svdvals(reshape(ψp, m, n)) ./ nrm
    p = abs2.(s)
    p = p[p .> 1e-14]
    return -sum(p .* log.(p))
end

function _disjoint_sets(named_sets::Pair{Symbol,<:AbstractVector{<:Integer}}...)
    seen = Set{Int}()
    for (name, xs) in named_sets
        for x in xs
            if x in seen
                throw(ArgumentError("site $x appears in multiple sets; sets must be disjoint"))
            end
            push!(seen, x)
        end
    end
    return true
end

"""
    block_mutual_information(ψ, A, B) -> Float64

Quantum mutual information `I(A:B) = S(A)+S(B)-S(A∪B)` for disjoint site sets.
"""
function block_mutual_information(ψ::AbstractArray, A::AbstractVector{<:Integer}, B::AbstractVector{<:Integer})
    _disjoint_sets(:A => A, :B => B)
    AB = [collect(A); collect(B)]
    return subsystem_entropy(ψ, A) + subsystem_entropy(ψ, B) - subsystem_entropy(ψ, AB)
end

"""
    conditional_mutual_information(ψ, A, B, C) -> Float64

Conditional mutual information `I(A:B|C) = S(A∪C)+S(B∪C)-S(C)-S(A∪B∪C)`.
"""
function conditional_mutual_information(ψ::AbstractArray,
    A::AbstractVector{<:Integer}, B::AbstractVector{<:Integer}, C::AbstractVector{<:Integer})
    _disjoint_sets(:A => A, :B => B, :C => C)
    AC = [collect(A); collect(C)]
    BC = [collect(B); collect(C)]
    ABC = [collect(A); collect(B); collect(C)]
    return subsystem_entropy(ψ, AC) + subsystem_entropy(ψ, BC) - subsystem_entropy(ψ, C) - subsystem_entropy(ψ, ABC)
end

"""
    interaction_information(ψ, A, B, C) -> Float64

Tripartite interaction information `I(A:B) - I(A:B|C)`. Negative values indicate
XOR-like synergy: `A` and `B` are more informative once `C` is known than they
are marginally.
"""
function interaction_information(ψ::AbstractArray,
    A::AbstractVector{<:Integer}, B::AbstractVector{<:Integer}, C::AbstractVector{<:Integer})
    return block_mutual_information(ψ, A, B) - conditional_mutual_information(ψ, A, B, C)
end

"""
    synergy_information(ψ, target, sources...) -> Float64

A simple non-additivity/synergy diagnostic:
`I(target : union(sources)) - Σ_i I(target : sources[i])`.
Positive values mean the sources jointly explain the target more than the sum of
pairwise explanations. This is not a full partial-information decomposition, but
it detects the failure mode of pairwise MI for XOR-like or implicit constraints.
"""
function synergy_information(ψ::AbstractArray, target::AbstractVector{<:Integer}, sources::AbstractVector{<:Integer}...)
    isempty(sources) && return 0.0
    _disjoint_sets((Symbol(:set, i) => s for (i, s) in enumerate((target, sources...)))...)
    allsources = reduce(vcat, [collect(s) for s in sources])
    joint = block_mutual_information(ψ, target, allsources)
    pair_sum = sum(block_mutual_information(ψ, target, s) for s in sources)
    return joint - pair_sum
end

"""
    total_correlation(ψ, sets...) -> Float64

Multi-information / total correlation: `Σ_i S(A_i) - S(∪_i A_i)` for disjoint
sets. Measures total dependence among all sets, but does not distinguish
redundancy from synergy.
"""
function total_correlation(ψ::AbstractArray, sets::AbstractVector{<:Integer}...)
    isempty(sets) && return 0.0
    _disjoint_sets((Symbol(:set, i) => s for (i, s) in enumerate(sets))...)
    union_sets = reduce(vcat, [collect(s) for s in sets])
    return sum(subsystem_entropy(ψ, s) for s in sets) - subsystem_entropy(ψ, union_sets)
end

"""
    block_mi_matrix(ψ, groups) -> Matrix{Float64}

Pairwise block-MI matrix for a vector of disjoint site groups.
"""
function block_mi_matrix(ψ::AbstractArray, groups::AbstractVector{<:AbstractVector{<:Integer}})
    n = length(groups)
    M = zeros(Float64, n, n)
    for i in 1:n, j in (i + 1):n
        M[i, j] = M[j, i] = block_mutual_information(ψ, groups[i], groups[j])
    end
    return M
end

# -----------------------------------------------------------------------------
# Classical / measurement-distribution versions
# -----------------------------------------------------------------------------
# These use p(x)=|ψ(x)|² / ||ψ||² and ordinary Shannon entropies of marginals.
# They are useful for detecting XOR-like classical synergy and for comparing
# against quantum block-entanglement diagnostics.

"""
    classical_subsystem_entropy(ψ, sites) -> Float64

Shannon entropy of the marginal probability distribution on `sites`, with
`p(x)=|ψ(x)|² / ||ψ||²`.
"""
function classical_subsystem_entropy(ψ::AbstractArray{T,L}, sites::AbstractVector{<:Integer}) where {T,L}
    A = sort(collect(Int, sites))
    all(1 .<= A .<= L) || throw(ArgumentError("sites must lie in 1:$L"))
    length(unique(A)) == length(A) || throw(ArgumentError("sites contains duplicates"))
    p = abs2.(ψ)
    z = sum(p)
    iszero(z) && throw(ArgumentError("cannot compute entropy of zero tensor"))
    p ./= z
    isempty(A) && return 0.0
    B = setdiff(collect(1:L), A)
    perm = [A; B]
    pp = permutedims(p, perm)
    dims = size(pp)
    m = prod(dims[1:length(A)])
    n = length(A) == L ? 1 : prod(dims[(length(A)+1):end])
    marg = vec(sum(reshape(pp, m, n); dims=2))
    marg = marg[marg .> 1e-14]
    return -sum(marg .* log.(marg))
end

function classical_block_mutual_information(ψ::AbstractArray, A::AbstractVector{<:Integer}, B::AbstractVector{<:Integer})
    _disjoint_sets(:A => A, :B => B)
    AB = [collect(A); collect(B)]
    return classical_subsystem_entropy(ψ, A) + classical_subsystem_entropy(ψ, B) - classical_subsystem_entropy(ψ, AB)
end

function classical_conditional_mutual_information(ψ::AbstractArray,
    A::AbstractVector{<:Integer}, B::AbstractVector{<:Integer}, C::AbstractVector{<:Integer})
    _disjoint_sets(:A => A, :B => B, :C => C)
    AC = [collect(A); collect(C)]
    BC = [collect(B); collect(C)]
    ABC = [collect(A); collect(B); collect(C)]
    return classical_subsystem_entropy(ψ, AC) + classical_subsystem_entropy(ψ, BC) -
           classical_subsystem_entropy(ψ, C) - classical_subsystem_entropy(ψ, ABC)
end

function classical_synergy_information(ψ::AbstractArray, target::AbstractVector{<:Integer}, sources::AbstractVector{<:Integer}...)
    isempty(sources) && return 0.0
    _disjoint_sets((Symbol(:set, i) => s for (i, s) in enumerate((target, sources...)))...)
    allsources = reduce(vcat, [collect(s) for s in sources])
    joint = classical_block_mutual_information(ψ, target, allsources)
    pair_sum = sum(classical_block_mutual_information(ψ, target, s) for s in sources)
    return joint - pair_sum
end

function classical_block_mi_matrix(ψ::AbstractArray, groups::AbstractVector{<:AbstractVector{<:Integer}})
    n = length(groups)
    M = zeros(Float64, n, n)
    for i in 1:n, j in (i + 1):n
        M[i, j] = M[j, i] = classical_block_mutual_information(ψ, groups[i], groups[j])
    end
    return M
end
