using Pkg
Pkg.activate(@__DIR__)

using LinearAlgebra
using Statistics
using Printf
using Random
import QuanticsGrids as QG
import MutualInformation as MI

# -----------------------------------------------------------------------------
# Experiment 1: 1D multi-scale function, everything exact.
#
# Question: does an MI-guided bit layout (MinLA on the MI matrix) reduce MPS
# bond dimensions? We validate two things:
#   (a) The MinLA objective ∑ W[π(i),π(j)]·|π(i)-π(j)| correlates with the
#       *actual* bond dimensions / cut entanglement -> justifying it as a proxy.
#   (b) The MI-optimal ordering beats random orderings (and is competitive
#       with the natural multi-scale quantics ordering, which is known to be
#       near-optimal for smooth 1D functions).
# -----------------------------------------------------------------------------

R = 10
grid = QG.DiscretizedGrid(R, -1.0, +1.0)
f(x) = sin(20π * x) * exp(-x^2)   # multi-scale: oscillations under Gaussian envelope
qf(q) = f(QG.quantics_to_origcoord(grid, q))

@info "R=$R  grid points = $(2^R)"

# --- exact MI matrix (R=10 is small enough for the exact method) ---
@time W = MI.mutualinformation(qf, R; method=:exact)
ψ = MI.amplitude_tensor(qf, fill(2, R))
ψ ./= norm(ψ)

# --- candidate orderings ---
natural = collect(1:R)
Random.seed!(1)
@time perm_mi, cost_mi, _ = MI.mi_ordering(W; max_iter=100_000, initial_temp=1.0, final_temp=1e-4)

# baseline: a batch of random orderings
Random.seed!(2)
n_rand = 60
rand_perms = [shuffle(1:R) for _ in 1:n_rand]

# --- evaluate every ordering on the *true* figures of merit ---
function evaluate(perm)
    χ = MI.bond_dimensions(ψ, perm)
    S = MI.cut_entropies(ψ, perm)
    return (perm=perm, minla=MI.minla_cost(W, perm),
            χmax=maximum(χ), χsum=sum(χ), smax=maximum(S), ssum=sum(S), χ=χ, S=S)
end

res_nat = evaluate(natural)
res_mi  = evaluate(perm_mi)
res_rand = evaluate.(rand_perms)

# --- summary table ---
function fmt_row(name, r)
    @printf("  %-22s  minla=%9.3f  χmax=%2d  χsum=%3d  Smax=%.4f  Ssum=%.4f  perm=%s\n",
        name, r.minla, r.χmax, r.χsum, r.smax, r.ssum, r.perm)
end

println("\n=== Orderings vs. true bond dimensions (R=$R) ===")
fmt_row("natural (1..R)", res_nat)
fmt_row("MI-optimal (MinLA)", res_mi)
χmax_rand = [r.χmax for r in res_rand]
smax_rand = [r.smax for r in res_rand]
@printf("  %-22s  χmax ∈ [%d, %d] (mean %.1f)   Smax ∈ [%.3f, %.3f]\n",
    "random (n=$n_rand)", minimum(χmax_rand), maximum(χmax_rand), mean(χmax_rand),
    minimum(smax_rand), maximum(smax_rand))
@printf("  %-22s  χmax ∈ [%d, %d] (mean %.1f)   Ssum ∈ [%.3f, %.3f]\n",
    "random (n=$n_rand)", minimum(χmax_rand), maximum(χmax_rand), mean(χmax_rand),
    minimum([r.ssum for r in res_rand]), maximum([r.ssum for r in res_rand]))

# --- (a) Does MinLA cost predict bond dimensions? ---
minla_all = [res_nat.minla, res_mi.minla, [r.minla for r in res_rand]...]
χmax_all  = [res_nat.χmax,  res_mi.χmax,  [r.χmax  for r in res_rand]...]
smax_all  = [res_nat.smax,  res_mi.smax,  [r.smax  for r in res_rand]...]
ssum_all  = [res_nat.ssum,  res_mi.ssum,  [r.ssum  for r in res_rand]...]

corr(a, b) = cor(a, b)
println("\n=== (a) Surrogate validity: Pearson corr(MinLA cost, figure of merit) ===")
@printf("  corr(MinLA, χmax)   = %+.3f\n", corr(minla_all, χmax_all))
@printf("  corr(MinLA, Smax)   = %+.3f\n", corr(minla_all, smax_all))
@printf("  corr(MinLA, Ssum)   = %+.3f\n", corr(minla_all, ssum_all))
@printf("  corr(Smax,  χmax)   = %+.3f   (entanglement vs rank, sanity)\n", corr(smax_all, χmax_all))

# --- (b) Did MI-optimal beat random? ---
n_rand_worse_χ = count(r.χmax > res_mi.χmax for r in res_rand)
n_rand_worse_s = count(r.smax > res_mi.smax for r in res_rand)
println("\n=== (b) MI-optimal vs random ===")
@printf("  random orderings with χmax > MI-optimal(=%d): %d / %d\n",
    res_mi.χmax, n_rand_worse_χ, n_rand)
@printf("  random orderings with Smax > MI-optimal(=%.4f): %d / %d\n",
    res_mi.smax, n_rand_worse_s, n_rand)

# bond-dimension profiles
println("\n=== Bond-dimension profiles χ_k along the chain ===")
println("  natural:    ", res_nat.χ)
println("  MI-optimal: ", res_mi.χ)
println("  best random:", res_rand[argmin([r.χmax for r in res_rand])].χ)

# save the MI matrix + profiles for later plotting (CairoMakie is broken on this build)
using DelimitedFiles
writedlm(joinpath(@__DIR__, "out_1d_MI_matrix.csv"), W, ',')
open(joinpath(@__DIR__, "out_1d_summary.txt"), "w") do io
    println(io, "R=$R  f=sin(20πx)exp(-x²)")
    println(io, "natural    χ=", res_nat.χ, " χmax=", res_nat.χmax, " Smax=", res_nat.smax, " minla=", res_nat.minla)
    println(io, "MI-optimal χ=", res_mi.χ, " χmax=", res_mi.χmax, " Smax=", res_mi.smax, " minla=", res_mi.minla, " perm=", res_mi.perm)
    println(io, "corr(MinLA,χmax)=", corr(minla_all,χmax_all))
    println(io, "corr(MinLA,Smax)=", corr(minla_all,smax_all))
end
@info "wrote out_1d_MI_matrix.csv, out_1d_summary.txt"
