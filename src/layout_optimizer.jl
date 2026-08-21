# Structured quantics bit-layout search.
#
# Pairwise-MI MinLA is a useful surrogate when correlations are pairwise, but it
# is not the actual objective. Bond dimensions are ranks of TT unfoldings across
# cuts. This file provides a small, dependency-light optimizer core:
#   1. describe bit groups/coordinates with BitLayoutSpec;
#   2. generate structured layout candidates (blocks, interleavings, sandwiches);
#   3. compute cheap pairwise-MI surrogate objectives;
#   4. optionally evaluate candidates against exact tensors via SVD;
#   5. generate data-driven layouts by beam search on sampled cut-SVD scores.
#
# TCI-based evaluation is intentionally kept in experiments/ so TensorCrossInterpolation
# does not become a hard dependency of the package.

"""
    BitLayoutSpec(groups::Dict{Symbol,<:AbstractVector{<:Integer}})

Description of a structured quantics bit layout problem. `groups` maps coordinate
or semantic group names (e.g. `:w`, `:kx`, `:ky`) to bit indices in natural
coarse-to-fine scale order. The groups must be disjoint and cover `1:R`.
"""
struct BitLayoutSpec
    groups::Dict{Symbol,Vector{Int}}
    group_order::Vector{Symbol}
    R::Int
end

function BitLayoutSpec(groups::Dict{Symbol,<:AbstractVector{<:Integer}}; group_order=nothing)
    order = group_order === nothing ? sort(collect(keys(groups)); by=string) : collect(Symbol.(group_order))
    all(haskey(groups, g) for g in order) || throw(ArgumentError("group_order contains groups not present in groups"))
    length(unique(order)) == length(order) || throw(ArgumentError("group_order contains duplicates"))
    length(order) == length(groups) || throw(ArgumentError("group_order must list every group exactly once"))

    gdict = Dict{Symbol,Vector{Int}}(g => collect(Int, groups[g]) for g in order)
    all(!isempty, values(gdict)) || throw(ArgumentError("groups must be non-empty"))
    bits = reduce(vcat, values(gdict))
    R = maximum(bits)
    sort(bits) == collect(1:R) || throw(ArgumentError("groups must be disjoint and cover 1:R"))
    return BitLayoutSpec(gdict, order, R)
end

"""
    LayoutCandidate(name, perm, family=:custom; metadata=Dict())

A named candidate bit ordering. `perm[k]` is the original bit placed at MPS chain
position `k`.
"""
struct LayoutCandidate
    name::String
    perm::Vector{Int}
    family::Symbol
    metadata::Dict{Symbol,Any}
end

LayoutCandidate(name::AbstractString, perm::AbstractVector{<:Integer}, family::Symbol=:custom;
    metadata=Dict{Symbol,Any}()) = LayoutCandidate(String(name), collect(Int, perm), family, Dict{Symbol,Any}(metadata))

function validate_candidate(spec::BitLayoutSpec, cand::LayoutCandidate)
    sort(cand.perm) == collect(1:spec.R) || throw(ArgumentError("candidate $(cand.name) is not a permutation of 1:$(spec.R)"))
    return true
end

canonical_perm_key(perm::AbstractVector{<:Integer}) = Tuple(perm)

function _coord_permutations(xs::Vector{Symbol})
    length(xs) == 1 && return [copy(xs)]
    out = Vector{Vector{Symbol}}()
    for i in eachindex(xs)
        rest = [xs[j] for j in eachindex(xs) if j != i]
        for p in _coord_permutations(rest)
            push!(out, [xs[i]; p])
        end
    end
    return out
end

function _push_unique!(out::Vector{LayoutCandidate}, seen::Set{Tuple{Vararg{Int}}}, cand::LayoutCandidate)
    key = canonical_perm_key(cand.perm)
    if !(key in seen)
        push!(out, cand)
        push!(seen, key)
    end
    return out
end

function _group_bits(spec::BitLayoutSpec, g::Symbol; reverse::Bool=false)
    bits = spec.groups[g]
    return reverse ? reverse!(copy(bits)) : copy(bits)
end

"""
    coordinate_block_layouts(spec; all_group_orders=true, reverse_blocks=true)

Generate layouts formed by concatenating whole coordinate/group blocks, optionally
trying all block orders and per-block reversals.
"""
function coordinate_block_layouts(spec::BitLayoutSpec; all_group_orders::Bool=true, reverse_blocks::Bool=true)
    orders = all_group_orders ? _coord_permutations(spec.group_order) : [spec.group_order]
    out = LayoutCandidate[]
    seen = Set{Tuple{Vararg{Int}}}()
    for order in orders
        nmask = reverse_blocks ? 2^length(order) : 1
        for mask in 0:(nmask - 1)
            parts = Vector{Vector{Int}}()
            revflags = Bool[]
            for (i, g) in enumerate(order)
                rev = reverse_blocks && (((mask >> (i - 1)) & 1) == 1)
                push!(parts, _group_bits(spec, g; reverse=rev))
                push!(revflags, rev)
            end
            perm = reduce(vcat, parts)
            suffix = reverse_blocks ? "_" * join([r ? "r" : "f" for r in revflags], "") : ""
            name = "block_" * join(order, "-") * suffix
            _push_unique!(out, seen, LayoutCandidate(name, perm, :block; metadata=Dict(:order => order, :reversed => revflags)))
        end
    end
    return out
end

"""
    scale_interleavings(spec; all_group_orders=true, scale_directions=true)

Generate Z-order/Morton-like layouts: for each scale, emit one bit from each
group in the selected group order. Requires all groups to have equal length.
"""
function scale_interleavings(spec::BitLayoutSpec; all_group_orders::Bool=true, scale_directions::Bool=true)
    lengths = length.(values(spec.groups))
    allequal(lengths) || throw(ArgumentError("scale interleavings require equal group lengths"))
    Rd = first(lengths)
    orders = all_group_orders ? _coord_permutations(spec.group_order) : [spec.group_order]
    dirs = scale_directions ? (1, -1) : (1,)
    out = LayoutCandidate[]
    seen = Set{Tuple{Vararg{Int}}}()
    for order in orders, dir in dirs
        scales = dir == 1 ? (1:Rd) : (Rd:-1:1)
        perm = [spec.groups[g][r] for r in scales for g in order]
        name = "interleave_" * join(order, "-") * (dir == 1 ? "_coarse" : "_fine")
        _push_unique!(out, seen, LayoutCandidate(name, perm, :interleave; metadata=Dict(:order => order, :scale_dir => dir)))
    end
    return out
end

"""
    pair_interleave_layouts(spec; scale_directions=true)

For each pair of groups, interleave the pair by scale and place all remaining
groups as blocks before or after the pair. Includes both pair orders and both
scale directions. For exactly two groups this duplicates ordinary scale
interleavings, so it returns no candidates.
"""
function pair_interleave_layouts(spec::BitLayoutSpec; scale_directions::Bool=true)
    length(spec.group_order) >= 3 || return LayoutCandidate[]
    lengths = length.(values(spec.groups))
    allequal(lengths) || throw(ArgumentError("pair interleavings require equal group lengths"))
    Rd = first(lengths)
    dirs = scale_directions ? (1, -1) : (1,)
    out = LayoutCandidate[]
    seen = Set{Tuple{Vararg{Int}}}()
    gs = spec.group_order
    for a in 1:length(gs), b in (a + 1):length(gs)
        pair0 = [gs[a], gs[b]]
        rest0 = setdiff(gs, pair0)
        rest_orders = _coord_permutations(rest0)
        for pair in (pair0, reverse(pair0)), rest_order in rest_orders, dir in dirs, rest_pos in (:before, :after)
            scales = dir == 1 ? (1:Rd) : (Rd:-1:1)
            pairpart = [spec.groups[g][r] for r in scales for g in pair]
            restparts = [_group_bits(spec, g; reverse=(dir == -1)) for g in rest_order]
            restpart = isempty(restparts) ? Int[] : reduce(vcat, restparts)
            perm = rest_pos === :before ? [restpart; pairpart] : [pairpart; restpart]
            name = "pair_" * join(pair, "-") * "_rest-" * join(rest_order, "-") * "_" * string(rest_pos) * (dir == 1 ? "_coarse" : "_fine")
            _push_unique!(out, seen, LayoutCandidate(name, perm, :pair_interleave;
                metadata=Dict(:pair => pair, :rest => rest_order, :rest_pos => rest_pos, :scale_dir => dir)))
        end
    end
    return out
end

"""
    sandwich_layouts(spec; middle_groups=spec.group_order, reverse_blocks=true)

Generate coordinate-block layouts with each group placed in the middle in turn.
For three groups this includes layouts like `[kx, reverse(w), reverse(ky)]`, the
family that worked well for the Green's function.
"""
function sandwich_layouts(spec::BitLayoutSpec; middle_groups=spec.group_order, reverse_blocks::Bool=true)
    length(spec.group_order) >= 3 || return LayoutCandidate[]
    out = LayoutCandidate[]
    seen = Set{Tuple{Vararg{Int}}}()
    for mid in middle_groups
        sides = setdiff(spec.group_order, [mid])
        for side_order in _coord_permutations(sides)
            order = [side_order[1], mid, side_order[2:end]...]
            nmask = reverse_blocks ? 2^length(order) : 1
            for mask in 0:(nmask - 1)
                parts = Vector{Vector{Int}}(); revflags = Bool[]
                for (i, g) in enumerate(order)
                    rev = reverse_blocks && (((mask >> (i - 1)) & 1) == 1)
                    push!(parts, _group_bits(spec, g; reverse=rev))
                    push!(revflags, rev)
                end
                perm = reduce(vcat, parts)
                name = "sandwich_" * join(order, "-") * "_" * join([r ? "r" : "f" for r in revflags], "")
                _push_unique!(out, seen, LayoutCandidate(name, perm, :sandwich;
                    metadata=Dict(:order => order, :middle => mid, :reversed => revflags)))
            end
        end
    end
    return out
end

"""
    generate_layout_candidates(spec; families=(:block,:interleave,:pair_interleave,:sandwich))

Generate a de-duplicated structured candidate set.
"""
function generate_layout_candidates(spec::BitLayoutSpec;
    families=(:block, :interleave, :pair_interleave, :sandwich))
    out = LayoutCandidate[]
    seen = Set{Tuple{Vararg{Int}}}()
    for fam in families
        cands = fam == :block ? coordinate_block_layouts(spec) :
                fam == :interleave ? scale_interleavings(spec) :
                fam == :pair_interleave ? pair_interleave_layouts(spec) :
                fam == :sandwich ? sandwich_layouts(spec) :
                throw(ArgumentError("unknown layout family :$fam"))
        for c in cands
            validate_candidate(spec, c)
            _push_unique!(out, seen, c)
        end
    end
    return out
end

"""
    cut_weights(W, perm) -> Vector{Float64}

Pairwise-MI crossing weight for every chain cut: `sum(W[i,j])` over pairs split
by that cut.
"""
function cut_weights(W::AbstractMatrix{<:Real}, perm::AbstractVector{<:Integer})
    n = length(perm)
    c = zeros(Float64, max(n - 1, 0))
    for k in 1:(n - 1)
        s = 0.0
        for i in 1:k, j in (k + 1):n
            s += W[perm[i], perm[j]]
        end
        c[k] = s
    end
    return c
end

cutwidth_objective(W::AbstractMatrix{<:Real}, perm::AbstractVector{<:Integer}) = begin
    cw = cut_weights(W, perm)
    (maxcut=isempty(cw) ? 0.0 : maximum(cw), sumcut=sum(cw))
end

layout_objectives(W::AbstractMatrix{<:Real}, cand::LayoutCandidate) = begin
    cw = cutwidth_objective(W, cand.perm)
    (minla=minla_cost(W, cand.perm), maxcut=cw.maxcut, sumcut=cw.sumcut)
end

struct ExactTensorLayoutEvaluator{A<:AbstractArray}
    ψ::A
    rtol::Float64
end
ExactTensorLayoutEvaluator(ψ::AbstractArray; rtol::Real=1e-12) = ExactTensorLayoutEvaluator(ψ, Float64(rtol))

function evaluate_layout(evaluator::ExactTensorLayoutEvaluator, cand::LayoutCandidate)
    χ = bond_dimensions(evaluator.ψ, cand.perm; rtol=evaluator.rtol)
    S = cut_entropies(evaluator.ψ, cand.perm)
    return (candidate=cand, χ=χ, S=S,
            χmax=isempty(χ) ? 1 : maximum(χ), χsum=sum(χ),
            Smax=isempty(S) ? 0.0 : maximum(S), Ssum=sum(S))
end

"""
    search_layouts(candidates; W=nothing, evaluator=nothing, top_k=16)

Rank candidates by cheap MI surrogates and/or direct evaluator. If an evaluator is
provided, a diverse shortlist (top by MinLA, cutwidth, plus named baselines) is
evaluated and returned sorted by `(χmax, χsum, Ssum)`.
"""
function search_layouts(candidates::AbstractVector{LayoutCandidate}; W=nothing, evaluator=nothing, top_k::Int=16)
    isempty(candidates) && return []
    if evaluator === nothing
        W === nothing && throw(ArgumentError("provide W or evaluator"))
        scored = [(candidate=c, objectives=layout_objectives(W, c)) for c in candidates]
        return sort(scored, by=x -> (x.objectives.maxcut, x.objectives.minla, x.objectives.sumcut))[1:min(top_k, length(scored))]
    end

    shortlist = LayoutCandidate[]
    seen = Set{Tuple{Vararg{Int}}}()
    if W !== nothing
        objs = [(candidate=c, objectives=layout_objectives(W, c)) for c in candidates]
        for x in sort(objs, by=x -> x.objectives.minla)[1:min(top_k, length(objs))]
            _push_unique!(shortlist, seen, x.candidate)
        end
        for x in sort(objs, by=x -> (x.objectives.maxcut, x.objectives.sumcut))[1:min(top_k, length(objs))]
            _push_unique!(shortlist, seen, x.candidate)
        end
    end
    for c in candidates
        if length(shortlist) >= top_k
            break
        end
        _push_unique!(shortlist, seen, c)
    end

    evaluated = [evaluate_layout(evaluator, c) for c in shortlist]
    return sort(evaluated, by=x -> (x.χmax, x.χsum, x.Ssum))
end

# -----------------------------------------------------------------------------
# Data-driven cut-sketch layout search
# -----------------------------------------------------------------------------

function _bitmask(bits::AbstractVector{<:Integer})
    m = UInt128(0)
    for b in bits
        b >= 1 || throw(ArgumentError("bit indices must be positive"))
        b <= 128 || throw(ArgumentError("bitmask keys support at most 128 bits"))
        m |= UInt128(1) << (b - 1)
    end
    return m
end

function _assignments(nbits::Int, nsamples::Int, rng)
    nbits == 0 && return [Int[]]
    total = nbits <= 62 ? 2^nbits : typemax(Int)
    if total <= nsamples
        return [index_to_config(i, fill(2, nbits)) for i in 1:total]
    else
        return [rand(rng, 1:2, nbits) for _ in 1:nsamples]
    end
end

"""
    sketch_cut_score(f, R, leftbits; nrow=64, ncol=64, rng=default_rng(), rtol=1e-10)

Estimate the TT cut difficulty for the bipartition `leftbits | complement` by
sampling a submatrix of the unfolding of `f`. Returns a named tuple containing
sampled singular-spectrum entropy, effective rank `exp(entropy)`, numerical
rank, and the singular values.

If one side has at most `nrow`/`ncol` configurations, that side is enumerated
exactly; otherwise configurations are sampled uniformly.
"""
function sketch_cut_score(f, R::Int, leftbits::AbstractVector{<:Integer};
    nrow::Int=64,
    ncol::Int=64,
    rng=default_rng(),
    rtol::Real=1e-10)
    left = sort(collect(Int, leftbits))
    all(1 .<= left .<= R) || throw(ArgumentError("leftbits must lie in 1:R"))
    length(unique(left)) == length(left) || throw(ArgumentError("leftbits contains duplicates"))
    right = setdiff(collect(1:R), left)
    if isempty(left) || isempty(right)
        return (entropy=0.0, effrank=1.0, rank=1, singular_values=[1.0], nrow=1, ncol=1)
    end

    rows = _assignments(length(left), nrow, rng)
    cols = _assignments(length(right), ncol, rng)
    A = Matrix{ComplexF64}(undef, length(rows), length(cols))
    q = ones(Int, R)
    for (i, rconf) in enumerate(rows)
        @inbounds for (p, b) in enumerate(left)
            q[b] = rconf[p]
        end
        for (j, cconf) in enumerate(cols)
            @inbounds for (p, b) in enumerate(right)
                q[b] = cconf[p]
            end
            A[i, j] = f(q)
        end
    end
    s = svdvals(A)
    if isempty(s) || iszero(sum(abs2, s))
        return (entropy=0.0, effrank=1.0, rank=0, singular_values=s, nrow=size(A,1), ncol=size(A,2))
    end
    p = abs2.(s) ./ sum(abs2, s)
    p = p[p .> 1e-14]
    entropy = -sum(p .* log.(p))
    rank = count(>(rtol * s[1]), s)
    return (entropy=entropy, effrank=exp(entropy), rank=rank, singular_values=s,
            nrow=size(A, 1), ncol=size(A, 2))
end

"""
    beam_search_layout_by_cut_sketch(f, R; beam_width=16, nrow=64, ncol=64, rng=default_rng())

Generate layouts directly from function data. The search builds a chain from
left to right. Each candidate prefix defines an actual TT cut; that cut is scored
by `sketch_cut_score`. Beam states are ranked lexicographically by
`(max_cut_entropy_so_far, sum_cut_entropy_so_far)`. The result is a list of
`LayoutCandidate`s with sketch profiles in metadata.

This does not assume coordinate groups or hand-written layout families.
"""
function beam_search_layout_by_cut_sketch(f, R::Int;
    beam_width::Int=16,
    nrow::Int=64,
    ncol::Int=64,
    rng=default_rng(),
    rtol::Real=1e-10)
    R >= 1 || throw(ArgumentError("R must be positive"))
    beam_width >= 1 || throw(ArgumentError("beam_width must be positive"))

    score_cache = Dict{UInt128,Any}()
    function score_for(prefix::Vector{Int})
        key = _bitmask(prefix)
        return get!(score_cache, key) do
            sketch_cut_score(f, R, prefix; nrow, ncol, rng, rtol)
        end
    end

    states = [(perm=Int[], remaining=collect(1:R), maxentropy=0.0, sumentropy=0.0,
               profile=Float64[], ranks=Int[])]
    for depth in 1:(R - 1)
        expanded = []
        for st in states
            for b in st.remaining
                prefix = [st.perm; b]
                sc = score_for(prefix)
                rem = [x for x in st.remaining if x != b]
                profile = [st.profile; sc.entropy]
                ranks = [st.ranks; sc.rank]
                push!(expanded, (perm=prefix, remaining=rem,
                    maxentropy=max(st.maxentropy, sc.entropy),
                    sumentropy=st.sumentropy + sc.entropy,
                    profile=profile, ranks=ranks))
            end
        end
        sort!(expanded, by=x -> (x.maxentropy, x.sumentropy))
        states = eltype(states)[]
        seen = Set{UInt128}()
        for st in expanded
            key = _bitmask(st.perm)
            if !(key in seen)
                push!(states, st)
                push!(seen, key)
                length(states) >= beam_width && break
            end
        end
    end

    cands = LayoutCandidate[]
    for (i, st) in enumerate(states)
        fullperm = [st.perm; st.remaining]
        push!(cands, LayoutCandidate("cutsketch_beam_$i", fullperm, :cutsketch_beam;
            metadata=Dict(:maxentropy => st.maxentropy, :sumentropy => st.sumentropy,
                          :entropy_profile => st.profile, :rank_profile => st.ranks,
                          :nrow => nrow, :ncol => ncol)))
    end
    return cands
end
