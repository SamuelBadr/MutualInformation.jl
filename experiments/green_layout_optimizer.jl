using Pkg; Pkg.activate(@__DIR__)
using Printf, Random, DelimitedFiles, Statistics
import QuanticsGrids as QG
import MutualInformation as MI
import TensorCrossInterpolation
include(joinpath(@__DIR__, "tci_layout_evaluator.jl"))

# Systematic Green's-function layout optimizer:
# 1. generate structured candidates from a BitLayoutSpec;
# 2. add pairwise-MI MinLA candidate;
# 3. screen with MI cutwidth/MinLA objectives;
# 4. evaluate a diverse shortlist by actual TCI bond dimensions.

Rd = 6
R = 3Rd
δ = 0.2
n_samples = 200_000

groups = Dict(:w => collect(1:Rd), :kx => collect((Rd+1):(2Rd)), :ky => collect((2Rd+1):(3Rd)))
spec = MI.BitLayoutSpec(groups; group_order=[:w, :kx, :ky])

w_inds = [[(:w, r)] for r in 1:Rd]
x_inds = [[(:kx, r)] for r in 1:Rd]
y_inds = [[(:ky, r)] for r in 1:Rd]
grid = QG.DiscretizedGrid((:w, :kx, :ky), [w_inds; x_inds; y_inds];
    lower_bound=(-5.0, -1π, -1π), upper_bound=(5.0, 1π, 1π))
disp(k) = -2 * sum(cos, k)
f(k; δ=δ) = 1 / (k[1] - disp(k[2:end]) + im * δ)
qf(q) = f(QG.quantics_to_origcoord(grid, q))

Wpath = joinpath(@__DIR__, "out_2d_MI_matrix.csv")
W = isfile(Wpath) ? readdlm(Wpath, ',', Float64) : MI.mutualinformation(qf, R; method=:uniform, n_samples)

candidates = MI.generate_layout_candidates(spec)
# Add pairwise-MI MinLA candidate from the replacement solver.
mi_perm, mi_cost, _ = MI.mi_ordering(W; algorithm=:heuristic, n_restarts=64, rng=MersenneTwister(123))
push!(candidates, MI.LayoutCandidate("pairwise_MI_MinLA", mi_perm, :mi_minla))

# Always include baselines by exact name/permutation.
baselines = [
    MI.LayoutCandidate("baseline_blocked", collect(1:R), :baseline),
    MI.LayoutCandidate("baseline_interleaved", [g[r] for r in 1:Rd for g in (groups[:w], groups[:kx], groups[:ky])], :baseline),
    MI.LayoutCandidate("known_sandwich_family", [groups[:kx]; reverse(groups[:w]); reverse(groups[:ky])], :baseline),
]
append!(candidates, baselines)
# Deduplicate.
seen = Set{Tuple{Vararg{Int}}}(); unique_candidates = MI.LayoutCandidate[]
for c in candidates
    k = Tuple(c.perm)
    if !(k in seen)
        push!(unique_candidates, c); push!(seen, k)
    end
end
candidates = unique_candidates

# Surrogate shortlist: top by MinLA, top by cutwidth, all block/sandwich baselines.
objs = [(candidate=c, objectives=MI.layout_objectives(W, c)) for c in candidates]
short = MI.LayoutCandidate[]; seen = Set{Tuple{Vararg{Int}}}()
function addcand!(c)
    k = Tuple(c.perm)
    if !(k in seen)
        push!(short, c); push!(seen, k)
    end
end
for x in sort(objs, by=x -> x.objectives.minla)[1:min(16, length(objs))]
    addcand!(x.candidate)
end
for x in sort(objs, by=x -> (x.objectives.maxcut, x.objectives.sumcut))[1:min(16, length(objs))]
    addcand!(x.candidate)
end
for c in candidates
    (c.family in (:block, :sandwich, :baseline, :mi_minla)) && addcand!(c)
end

println("Generated $(length(candidates)) candidates; TCI-evaluating $(length(short)) candidates.")
evaluator = TCILayoutEvaluator(qf, R; tolerance=1e-8, maxiter=100,
    pivots=[ones(Int, R), fill(2, R)])
results = []
for c in short
    ev = MI.evaluate_layout(evaluator, c)
    obj = MI.layout_objectives(W, c)
    push!(results, merge(ev, (minla=obj.minla, maxcut=obj.maxcut, sumcut=obj.sumcut)))
    @printf("%-40s %-14s χmax=%3d χsum=%4d minla=%7.3f maxcut=%6.3f err=%.1e\n",
        c.name[1:min(end,40)], string(c.family), ev.χmax, ev.χsum, obj.minla, obj.maxcut, ev.final_error)
end

println("\nBest layouts by actual TCI χmax:")
for r in sort(results, by=x -> (x.χmax, x.χsum))[1:min(12, length(results))]
    c = r.candidate
    @printf("%-40s %-14s χmax=%3d χsum=%4d minla=%7.3f maxcut=%6.3f perm=%s\n",
        c.name[1:min(end,40)], string(c.family), r.χmax, r.χsum, r.minla, r.maxcut, c.perm)
end
