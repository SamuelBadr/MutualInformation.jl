using Pkg; Pkg.activate(@__DIR__)
using LinearAlgebra, Statistics, Printf, Random, DelimitedFiles
import QuanticsGrids as QG
import TensorCrossInterpolation as TCI
import MutualInformation as MI

# Broader layout search for the Green's function after replacing the OCT backend.
# Tries structured block/interleaved/pair-interleaved layouts and a pairwise-MI
# cutwidth objective. Reports actual TCI bond dimensions.

Rd = 6
R = 3Rd
coords = (:w, :kx, :ky)
offset = Dict(:w => 0, :kx => Rd, :ky => 2Rd)
idx(c, r) = offset[c] + r

w_inds = [[(:w, r)] for r in 1:Rd]
x_inds = [[(:kx, r)] for r in 1:Rd]
y_inds = [[(:ky, r)] for r in 1:Rd]
indextable = [w_inds; x_inds; y_inds]
grid = QG.DiscretizedGrid((:w, :kx, :ky), indextable;
    lower_bound=(-5.0, -1π, -1π), upper_bound=(5.0, 1π, 1π))
disp(k) = -2 * sum(cos, k)
δ = 0.2
f(k; δ=δ) = 1 / (k[1] - disp(k[2:end]) + im * δ)
qf(q) = f(QG.quantics_to_origcoord(grid, q))

function coord_perms(xs)
    length(xs) == 1 && return [collect(xs)]
    out = Vector{Vector{eltype(xs)}}()
    for i in eachindex(xs)
        rest = [xs[j] for j in eachindex(xs) if j != i]
        for p in coord_perms(rest)
            push!(out, [xs[i]; p])
        end
    end
    return out
end

coord_orders = coord_perms(collect(coords))

block_perm(order; dirs=Dict(c => 1 for c in coords)) = reduce(vcat, [dirs[c] == 1 ? [idx(c, r) for r in (1:Rd)] : [idx(c, r) for r in (Rd:-1:1)] for c in order])
interleaved_perm(order; scale_dir=1) = [idx(c, r) for r in (scale_dir == 1 ? (1:Rd) : (Rd:-1:1)) for c in order]
pair_interleave_perm(pair, third; third_pos=:after, scale_dir=1) = begin
    rs = scale_dir == 1 ? (1:Rd) : (Rd:-1:1)
    pairpart = [idx(c, r) for r in rs for c in pair]
    thirdpart = [idx(third, r) for r in rs]
    third_pos == :before ? [thirdpart; pairpart] : [pairpart; thirdpart]
end

function cut_weights(W, perm)
    n = length(perm)
    cw = zeros(Float64, n - 1)
    for k in 1:n-1
        s = 0.0
        for a in 1:k, b in k+1:n
            s += W[perm[a], perm[b]]
        end
        cw[k] = s
    end
    return cw
end
cutwidth_cost(W, perm) = (maximum(cut_weights(W, perm)), sum(cut_weights(W, perm)))

function inserted(perm, i, j)
    v = perm[i]
    rest = [perm[k] for k in eachindex(perm) if k != i]
    j = min(j, length(perm))
    return [rest[1:j-1]; v; rest[j:end]]
end
function reversed_segment(perm, i, j)
    p = copy(perm); reverse!(@view p[i:j]); p
end
function improve_cutwidth(W, init; max_passes=20)
    perm = copy(init); cur = cutwidth_cost(W, perm)
    for _ in 1:max_passes
        bestp = perm; best = cur
        n = length(perm)
        for i in 1:n, j in 1:n
            (j == i || j == i + 1) && continue
            p = inserted(perm, i, j); c = cutwidth_cost(W, p)
            c < best && (best = c; bestp = p)
        end
        for i in 1:n-1, j in i+1:n
            p = reversed_segment(perm, i, j); c = cutwidth_cost(W, p)
            c < best && (best = c; bestp = p)
        end
        best < cur || break
        perm = bestp; cur = best
    end
    return perm, cur
end

# Use the most recently generated Green MI matrix if available; otherwise compute.
Wpath = joinpath(@__DIR__, "out_2d_MI_matrix.csv")
W = isfile(Wpath) ? readdlm(Wpath, ',', Float64) : MI.mutualinformation(qf, R; method=:uniform, n_samples=200_000)

candidates = Dict{String,Vector{Int}}()
# Coordinate blocks, all coordinate orders and forward/reverse within each coordinate.
for order in coord_orders
    for mask in 0:7
        dirs = Dict(coords[i] => ((mask >> (i - 1)) & 1 == 0 ? 1 : -1) for i in 1:3)
        name = "block_" * join(order, "-") * "_" * join([dirs[c] == 1 ? "f" : "r" for c in order], "")
        candidates[name] = block_perm(order; dirs)
    end
end
# Full scale interleavings.
for order in coord_orders, sd in (1, -1)
    candidates["interleave_" * join(order, "-") * (sd == 1 ? "_coarse" : "_fine")] = interleaved_perm(order; scale_dir=sd)
end
# Pair interleavings with the third coordinate blocked before/after.
for pair in ((:w, :kx), (:w, :ky), (:kx, :ky))
    for pair_order in (collect(pair), reverse(collect(pair)))
        third = only(setdiff(collect(coords), pair_order))
        for pos in (:before, :after), sd in (1, -1)
            candidates["pair_" * join(pair_order, "-") * "_" * string(third) * "_" * string(pos) * (sd == 1 ? "_coarse" : "_fine")] = pair_interleave_perm(pair_order, third; third_pos=pos, scale_dir=sd)
        end
    end
end

# Pairwise-MI cutwidth objective (minimize maximum pairwise MI crossing any cut).
rng = MersenneTwister(11)
starts = [collect(1:R), [idx(c, r) for c in (:w, :kx, :ky) for r in 1:Rd], interleaved_perm([:w, :kx, :ky])]
for _ in 1:20
    p = collect(1:R); shuffle!(rng, p); push!(starts, p)
end
function best_cutwidth_layout(W, starts)
    bestcw = (Inf, Inf)
    bestp = copy(first(starts))
    for s in starts
        p, c = improve_cutwidth(W, s)
        if c < bestcw
            bestcw = c
            bestp = p
        end
    end
    return bestp, bestcw
end
bestp, bestcw = best_cutwidth_layout(W, starts)
candidates["MI_cutwidth"] = bestp

function tci_dims(perm; tolerance=1e-8)
    invp = invperm(perm)
    g(u) = qf(u[invp])
    pivots = [ones(Int, R), fill(2, R)]
    tci, ranks, errors = TCI.crossinterpolate2(ComplexF64, g, fill(2, R), pivots; tolerance, maxiter=100)
    χ = [TCI.linkdim(tci, k) for k in 1:R-1]
    return χ, isempty(errors) ? NaN : errors[end]
end

# Screen by pairwise objectives first; evaluate a diverse subset + best by each objective.
objs = [(name=name, perm=p, minla=MI.minla_cost(W, p), cut=cutwidth_cost(W, p)) for (name, p) in candidates]
by_minla = sort(objs, by=x -> x.minla)[1:12]
by_cut = sort(objs, by=x -> x.cut)[1:12]
manual = [x for x in objs if startswith(x.name, "block_w-kx-ky_fff") || startswith(x.name, "interleave_w-kx-ky_coarse") || x.name == "MI_cutwidth"]
toeval = Dict{String,Vector{Int}}()
for x in vcat(by_minla, by_cut, manual)
    toeval[x.name] = x.perm
end

println("Evaluating $(length(toeval)) candidate layouts by actual TCI bond dimensions...")
results = []
for (name, perm) in toeval
    χ, err = tci_dims(perm)
    push!(results, (name=name, perm=perm, χmax=maximum(χ), χsum=sum(χ), err=err,
                    minla=MI.minla_cost(W, perm), cut=cutwidth_cost(W, perm)))
    @printf("%-45s χmax=%3d χsum=%4d minla=%7.3f cutmax=%6.3f err=%.1e\n",
        name, maximum(χ), sum(χ), MI.minla_cost(W, perm), cutwidth_cost(W, perm)[1], err)
end

println("\nBest by χmax:")
for r in sort(results, by=x -> (x.χmax, x.χsum))[1:min(10, length(results))]
    @printf("%-45s χmax=%3d χsum=%4d minla=%7.3f cutmax=%6.3f perm=%s\n",
        r.name, r.χmax, r.χsum, r.minla, r.cut[1], r.perm)
end
