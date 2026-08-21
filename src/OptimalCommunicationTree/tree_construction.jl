using LinearAlgebra: Diagonal, Symmetric, eigen
using Random: AbstractRNG, default_rng, shuffle!
using Graphs

# -----------------------------------------------------------------------------
# Minimum Linear Arrangement (MinLA) path solver
# -----------------------------------------------------------------------------
# For MPS/quantics bit layouts the relevant OCT case is max_deg == 2. A spanning
# tree with maximum degree two is a Hamiltonian path, and the OCT objective
#
#     ∑_{i<j} W[i,j] dist_T(i,j)
#
# becomes exactly the weighted Minimum Linear Arrangement objective
#
#     ∑_{i<j} W[perm[i], perm[j]] (j - i).
#
# This file therefore treats max_deg == 2 as the precise, first-class problem:
# exact exhaustive search for small n, deterministic multi-start local search for
# larger n. General degree-bounded OCT is intentionally not approximated here; if
# callers request max_deg != 2 they get a clear error instead of an untrusted tree
# heuristic.

function _validate_weight_matrix(W::AbstractMatrix{<:Real}; check_symmetric::Bool=true)
    n = size(W, 1)
    size(W, 2) == n || throw(ArgumentError("W must be square; got size $(size(W))"))
    all(isfinite, W) || throw(ArgumentError("W must contain only finite entries"))
    all(w -> w >= 0, W) || throw(ArgumentError("W must have non-negative entries"))
    if check_symmetric && !isapprox(W, W'; rtol=1e-10, atol=1e-12)
        throw(ArgumentError("W must be symmetric for an undirected layout problem"))
    end
    return n
end

function _validate_permutation(perm::AbstractVector{<:Integer}, n::Int)
    length(perm) == n || throw(ArgumentError("perm length $(length(perm)) does not match n=$n"))
    sort(collect(perm)) == collect(1:n) || throw(ArgumentError("perm must be a permutation of 1:$n"))
    return true
end

"""
    minla_cost(W, perm) -> Float64

Weighted Minimum Linear Arrangement objective for the site ordering `perm`:
`∑_{i<j} W[perm[i], perm[j]] * (j - i)`.
"""
function minla_cost(W::AbstractMatrix{<:Real}, perm::AbstractVector{<:Integer})
    n = _validate_weight_matrix(W)
    _validate_permutation(perm, n)
    total = 0.0
    @inbounds for i in 1:n
        pi = perm[i]
        for j in (i + 1):n
            total += W[pi, perm[j]] * (j - i)
        end
    end
    return total
end

"""
    path_graph_from_ordering(perm) -> SimpleGraph

Construct the Hamiltonian path graph whose chain order is `perm`.
"""
function path_graph_from_ordering(perm::AbstractVector{<:Integer})
    n = length(perm)
    _validate_permutation(perm, n)
    tree = SimpleGraph(n)
    for k in 1:(n - 1)
        add_edge!(tree, perm[k], perm[k + 1])
    end
    return tree
end

"""
    ordering_from_path(tree::SimpleGraph) -> Vector{Int}

Recover a linear ordering from a connected path graph. Throws if `tree` is not a
Hamiltonian path (except for the trivial 0- and 1-node cases).
"""
function ordering_from_path(tree::SimpleGraph)
    n = nv(tree)
    n == 0 && return Int[]
    n == 1 && return [1]
    ne(tree) == n - 1 || throw(ArgumentError("graph must have n-1 edges to be a path/tree"))
    is_connected(tree) || throw(ArgumentError("graph must be connected"))
    degs = [degree(tree, v) for v in 1:n]
    all(<=(2), degs) || throw(ArgumentError("graph maximum degree must be ≤ 2 to be a path"))
    endpoints = findall(==(1), degs)
    length(endpoints) == 2 || throw(ArgumentError("nontrivial path must have exactly two endpoints"))

    order = Int[endpoints[1]]
    visited = falses(n)
    visited[endpoints[1]] = true
    u = endpoints[1]
    while length(order) < n
        next = 0
        for v in neighbors(tree, u)
            if !visited[v]
                next = v
                break
            end
        end
        next == 0 && throw(ArgumentError("graph traversal ended before visiting all nodes"))
        push!(order, next)
        visited[next] = true
        u = next
    end
    return order
end

"""
    oct_cost(tree, W)

Compute `∑_{i<j} W[i,j] * dist_T(i,j)`. For path graphs this is identical to
`minla_cost(W, ordering_from_path(tree))`.
"""
function oct_cost(tree::SimpleGraph, W::AbstractMatrix{<:Real})
    n = _validate_weight_matrix(W)
    nv(tree) == n || throw(ArgumentError("tree has $(nv(tree)) vertices but W is $n×$n"))
    if n <= 1
        return 0.0
    end
    ne(tree) == n - 1 || throw(ArgumentError("graph must have n-1 edges to be a spanning tree"))
    is_connected(tree) || throw(ArgumentError("graph must be connected"))

    total = 0.0
    for src in 1:n
        distances = gdistances(tree, src)
        for j in (src + 1):n
            total += W[src, j] * distances[j]
        end
    end
    return total
end

"""
    exact_minla(W) -> (perm, cost)

Globally optimal MinLA solution by exhaustive enumeration. Intended for small
`n`; `solve_minla(...; algorithm=:auto)` uses this only up to
`exact_threshold`.
"""
function exact_minla(W::AbstractMatrix{<:Real}; verbose::Bool=false)
    n = _validate_weight_matrix(W)
    n == 0 && return Int[], 0.0
    n == 1 && return [1], 0.0

    perm = zeros(Int, n)
    used = falses(n)
    best_perm = collect(1:n)
    best_cost = Inf
    nvisited = 0

    function recurse!(pos::Int)
        if pos > n
            # Reversal symmetry: perm and reverse(perm) have identical cost.
            # Keep exactly one representative.
            if perm[1] > perm[end]
                return nothing
            end
            nvisited += 1
            c = minla_cost(W, perm)
            if c < best_cost
                best_cost = c
                best_perm = copy(perm)
            end
            return nothing
        end
        for v in 1:n
            if !used[v]
                perm[pos] = v
                used[v] = true
                recurse!(pos + 1)
                used[v] = false
            end
        end
        return nothing
    end

    recurse!(1)
    verbose && @info "exact_minla enumerated $nvisited reversal-distinct permutations" best_cost best_perm
    return best_perm, best_cost
end

function _spectral_ordering(W::AbstractMatrix{<:Real})
    n = size(W, 1)
    n <= 1 && return collect(1:n)
    d = vec(sum(W; dims=2))
    if all(iszero, d)
        return collect(1:n)
    end
    L = Diagonal(d) - W
    F = eigen(Symmetric(Matrix(L)))
    # If the graph is disconnected, the Fiedler subspace is not unique. Sorting
    # by the second eigenvector is still a deterministic useful initializer.
    v = F.vectors[:, min(2, n)]
    return sortperm(v; alg=MergeSort)
end

function _heaviest_edge_ordering(W::AbstractMatrix{<:Real})
    n = size(W, 1)
    n <= 1 && return collect(1:n)
    best = (-Inf, 1, min(2, n))
    for i in 1:n, j in (i + 1):n
        if W[i, j] > best[1]
            best = (W[i, j], i, j)
        end
    end
    order = [best[2], best[3]]
    remaining = Set(1:n)
    delete!(remaining, best[2]); delete!(remaining, best[3])
    while !isempty(remaining)
        left = first(order); right = last(order)
        best_score = -Inf
        best_v = first(remaining)
        best_side = :right
        for v in remaining
            lscore = W[v, left]
            rscore = W[v, right]
            # Tie-break by total connection to current path.
            total_conn = sum(W[v, u] for u in order)
            if lscore + 1e-14 * total_conn > best_score
                best_score = lscore + 1e-14 * total_conn
                best_v = v; best_side = :left
            end
            if rscore + 1e-14 * total_conn > best_score
                best_score = rscore + 1e-14 * total_conn
                best_v = v; best_side = :right
            end
        end
        if best_side === :left
            pushfirst!(order, best_v)
        else
            push!(order, best_v)
        end
        delete!(remaining, best_v)
    end
    return order
end

function _greedy_insertion_ordering(W::AbstractMatrix{<:Real})
    n = size(W, 1)
    n <= 2 && return collect(1:n)
    order = _heaviest_edge_ordering(W)[1:2]
    remaining = Set(1:n)
    foreach(v -> delete!(remaining, v), order)
    while !isempty(remaining)
        best_c = Inf
        best_order = Int[]
        for v in remaining
            for pos in 1:(length(order) + 1)
                cand = copy(order)
                insert!(cand, pos, v)
                c = minla_cost(@view(W[cand, cand]), collect(1:length(cand)))
                if c < best_c
                    best_c = c
                    best_order = cand
                end
            end
        end
        order = best_order
        for v in order
            delete!(remaining, v)
        end
    end
    return order
end

function _inserted(perm::Vector{Int}, i::Int, j::Int)
    # Remove position i and insert the removed element at position j in the final
    # length-n ordering. j ∈ 1:n. Some (i,j) pairs are no-ops.
    n = length(perm)
    v = perm[i]
    rest = Vector{Int}(undef, n - 1)
    k = 1
    @inbounds for p in 1:n
        if p != i
            rest[k] = perm[p]
            k += 1
        end
    end
    jrest = min(j, n)
    cand = Vector{Int}(undef, n)
    @inbounds begin
        for p in 1:(jrest - 1)
            cand[p] = rest[p]
        end
        cand[jrest] = v
        for p in jrest:(n - 1)
            cand[p + 1] = rest[p]
        end
    end
    return cand
end

function _reversed_segment(perm::Vector{Int}, i::Int, j::Int)
    cand = copy(perm)
    reverse!(@view cand[i:j])
    return cand
end

function _local_search_minla(W::AbstractMatrix{<:Real}, init::Vector{Int};
    max_passes::Int=typemax(Int), verbose::Bool=false)
    n = size(W, 1)
    perm = copy(init)
    current = minla_cost(W, perm)
    pass = 0
    improved = true

    while improved && pass < max_passes
        pass += 1
        improved = false
        best_perm = perm
        best_cost = current

        # Best insertion move.
        for i in 1:n, j in 1:n
            (j == i || j == i + 1) && continue
            cand = _inserted(perm, i, j)
            c = minla_cost(W, cand)
            if c < best_cost - 1e-12
                best_cost = c
                best_perm = cand
            end
        end

        # Best segment reversal / 2-opt move.
        for i in 1:(n - 1), j in (i + 1):n
            cand = _reversed_segment(perm, i, j)
            c = minla_cost(W, cand)
            if c < best_cost - 1e-12
                best_cost = c
                best_perm = cand
            end
        end

        if best_cost < current - 1e-12
            perm = best_perm
            current = best_cost
            improved = true
            verbose && @info "MinLA local-search improvement" pass current perm
        end
    end

    return perm, current
end

function _unique_push!(perms::Vector{Vector{Int}}, seen::Set{Tuple{Vararg{Int}}}, perm::Vector{Int})
    key = Tuple(perm)
    if !(key in seen)
        push!(perms, copy(perm))
        push!(seen, key)
    end
    return perms
end

"""
    heuristic_minla(W; n_restarts=32, rng=default_rng(), max_passes=typemax(Int))

Deterministic/restarted local-search heuristic for larger MinLA instances. It
starts from several auditable initializers (identity, spectral/Fiedler, greedy
path growth, greedy insertion, and random restarts) and monotonically improves
with insertion and segment-reversal moves. The returned solution is never worse
than the best initializer because every accepted move strictly decreases the
objective.
"""
function heuristic_minla(W::AbstractMatrix{<:Real};
    n_restarts::Int=32,
    rng::AbstractRNG=default_rng(),
    max_passes::Int=typemax(Int),
    verbose::Bool=false)
    n = _validate_weight_matrix(W)
    n <= 1 && return collect(1:n), 0.0

    initializers = Vector{Vector{Int}}()
    seen = Set{Tuple{Vararg{Int}}}()
    for p in (collect(1:n), reverse(collect(1:n)), _spectral_ordering(W),
              reverse(_spectral_ordering(W)), _heaviest_edge_ordering(W),
              _greedy_insertion_ordering(W))
        _unique_push!(initializers, seen, collect(p))
    end
    for _ in 1:n_restarts
        p = collect(1:n)
        shuffle!(rng, p)
        _unique_push!(initializers, seen, p)
    end

    best_perm = first(initializers)
    best_cost = minla_cost(W, best_perm)
    for init in initializers
        perm, cost = _local_search_minla(W, init; max_passes, verbose=false)
        if cost < best_cost - 1e-12
            best_perm = perm
            best_cost = cost
            verbose && @info "New best heuristic MinLA" best_cost best_perm
        end
    end
    return best_perm, best_cost
end

"""
    solve_minla(W; algorithm=:auto, exact_threshold=9, kwargs...) -> (perm, cost)

Solve weighted MinLA exactly for small `n` and heuristically for larger `n`.
`algorithm` may be `:auto`, `:exact`, or `:heuristic`.
"""
function solve_minla(W::AbstractMatrix{<:Real};
    algorithm::Symbol=:auto,
    exact_threshold::Int=9,
    n_restarts::Int=32,
    max_iter::Int=10_000,
    rng::AbstractRNG=default_rng(),
    verbose::Bool=false,
    kwargs...)
    n = _validate_weight_matrix(W)
    algorithm in (:auto, :exact, :heuristic) ||
        throw(ArgumentError("algorithm must be :auto, :exact, or :heuristic; got :$algorithm"))

    if algorithm == :exact || (algorithm == :auto && n <= exact_threshold)
        n <= exact_threshold || algorithm == :exact || error("unreachable")
        if algorithm == :exact && n > exact_threshold
            @warn "Exact MinLA requested for n=$n; exhaustive enumeration may be very slow"
        end
        return exact_minla(W; verbose)
    end

    max_passes = max(1, max_iter)
    return heuristic_minla(W; n_restarts, rng, max_passes, verbose)
end

"""
    solve_oct(W; max_deg=2, kwargs...) -> (tree, cost)

API-compatible OCT entry point. The supported and trusted case is
`max_deg == 2`, where OCT is exactly weighted MinLA / Hamiltonian path layout.
For other degree bounds this method throws an `ArgumentError` rather than
returning an untrusted heuristic tree.
"""
function solve_oct(W::AbstractMatrix{<:Real}; max_deg::Int=2, kwargs...)
    n = _validate_weight_matrix(W)
    if n <= 1
        return SimpleGraph(n), 0.0
    elseif n == 2
        tree = SimpleGraph(2)
        add_edge!(tree, 1, 2)
        return tree, W[1, 2]
    end

    max_deg == 2 || throw(ArgumentError(
        "This package now supports the trusted OCT/MinLA path case only (max_deg=2). " *
        "General degree-bounded OCT is not implemented because the previous tree heuristic was not reliable."))

    perm, cost = solve_minla(W; kwargs...)
    tree = path_graph_from_ordering(perm)
    # Cost equality is the core equivalence; keep the check as a guard against
    # future path/convention regressions.
    oc = oct_cost(tree, W)
    isapprox(oc, cost; rtol=1e-10, atol=1e-10) ||
        error("internal error: OCT path cost $oc does not match MinLA cost $cost")
    return tree, cost
end

"""
    solve_oct_problem(W, max_degree_bound; kwargs...)

Solve the trusted degree-2 OCT/MinLA path problem and return
`(tree, cost, edges)`. `max_degree_bound` must be 2 for nontrivial graphs.
"""
function solve_oct_problem(W::AbstractMatrix{<:Real}, max_degree_bound::Int; kwargs...)
    tree, cost = solve_oct(W; max_deg=max_degree_bound, kwargs...)
    edge_list = [(min(src(e), dst(e)), max(src(e), dst(e))) for e in edges(tree)]
    return (tree=tree, cost=cost, edges=edge_list)
end
