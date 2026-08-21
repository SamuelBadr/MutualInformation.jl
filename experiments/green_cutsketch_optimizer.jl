using Pkg; Pkg.activate(@__DIR__)
using Printf, Random, DelimitedFiles
import QuanticsGrids as QG
import MutualInformation as MI
import TensorCrossInterpolation
include(joinpath(@__DIR__, "tci_layout_evaluator.jl"))

# Data-driven Green layout search. Unlike green_layout_optimizer.jl, this does
# not start from coordinate-layout families. It builds bit orders by beam search
# on sampled SVDs of the actual TT cuts of qf.

Rd = 6
R = 3Rd
δ = 0.2
w_inds = [[(:w, r)] for r in 1:Rd]
x_inds = [[(:kx, r)] for r in 1:Rd]
y_inds = [[(:ky, r)] for r in 1:Rd]
grid = QG.DiscretizedGrid((:w, :kx, :ky), [w_inds; x_inds; y_inds];
    lower_bound=(-5.0, -1π, -1π), upper_bound=(5.0, 1π, 1π))
disp(k) = -2 * sum(cos, k)
f(k; δ=δ) = 1 / (k[1] - disp(k[2:end]) + im * δ)
qf(q) = f(QG.quantics_to_origcoord(grid, q))

println("Running cut-sketch beam search from function evaluations...")
@time beam = MI.beam_search_layout_by_cut_sketch(qf, R; beam_width=18, nrow=48, ncol=48,
    rng=MersenneTwister(2024), rtol=1e-9)
println("Generated $(length(beam)) data-driven layouts")
for c in beam[1:min(end, 8)]
    @printf("%-18s maxS=%.3f sumS=%.3f ranks=%s perm=%s\n", c.name,
        c.metadata[:maxentropy], c.metadata[:sumentropy], c.metadata[:rank_profile], c.perm)
end

# Compare against known baselines by actual TCI.
baselines = [
    MI.LayoutCandidate("blocked", collect(1:R), :baseline),
    MI.LayoutCandidate("interleaved", [r + off for r in 1:Rd for off in (0, Rd, 2Rd)], :baseline),
    MI.LayoutCandidate("sandwich", [collect((2Rd+1):(3Rd)); collect(Rd:-1:1); collect((Rd+1):(2Rd))], :baseline),
]
candidates = [beam; baselines]
evaluator = TCILayoutEvaluator(qf, R; tolerance=1e-8, maxiter=100,
    pivots=[ones(Int, R), fill(2, R)])

println("\nTCI validation:")
results = []
for c in candidates
    ev = MI.evaluate_layout(evaluator, c)
    push!(results, ev)
    @printf("%-18s %-14s χmax=%3d χsum=%4d err=%.1e perm=%s\n",
        c.name, string(c.family), ev.χmax, ev.χsum, ev.final_error, c.perm)
end

println("\nBest by χmax:")
for ev in sort(results, by=x -> (x.χmax, x.χsum))[1:min(end, 10)]
    @printf("%-18s %-14s χmax=%3d χsum=%4d perm=%s\n",
        ev.candidate.name, string(ev.candidate.family), ev.χmax, ev.χsum, ev.candidate.perm)
end
