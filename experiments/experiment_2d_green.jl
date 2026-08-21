using Pkg
Pkg.activate(@__DIR__)

using LinearAlgebra
using Statistics
using Printf
using Random
import QuanticsGrids as QG
import TensorCrossInterpolation as TCI
import MutualInformation as MI

# -----------------------------------------------------------------------------
# Experiment 2: 3D Green's function  G(w, kx, ky) = 1 / (w - ε(kx,ky) + iδ)
#   with ε(k) = -2 cos(kx) - 2 cos(ky)  (2D tight-binding dispersion).
#
# This is the case where the *bit interleaving* of the three coordinates
# genuinely controls the quantics-TT bond dimension: the pole lives on the 2D
# manifold w = ε(kx,ky), coupling the w bits to the (kx,ky) bits. We compare
# three layouts of the R bits on the MPS chain:
#   - blocked      : all w bits, then all kx bits, then all ky bits  (the
#                    layout used in the original scripts/plot_mutual_info_2D.jl)
#   - interleaved  : Z-order by scale: w_r, kx_r, ky_r for r = 1..Rd
#   - MI-optimal    : Hamiltonian path from the MI matrix (MinLA on pairwise MI)
# ... and measure the ACTUAL TT bond dimensions via tensor cross interpolation.
# -----------------------------------------------------------------------------

Rd = 6                                  # bits per coordinate
Rk, Rnu, Rw = Rd, Rd, Rd
R = Rk + Rnu + Rw                       # = 18 bits total
w_inds = [[(:w, r)]  for r in 1:Rw]
x_inds = [[(:kx, r)] for r in 1:Rk]
y_inds = [[(:ky, r)] for r in 1:Rk]
indextable = [w_inds; x_inds; y_inds]   # blocked layout (grid-natural ordering)
wmax = 5.0
grid = QG.DiscretizedGrid((:w, :kx, :ky), indextable;
    lower_bound=(-wmax, -1π, -1π), upper_bound=(+wmax, +1π, +1π))

disp(k) = -2 * sum(cos, k)
δ = 0.2
f(k; δ=δ) = 1 / (k[1] - disp(k[2:end]) + im * δ)
qf(q) = f(QG.quantics_to_origcoord(grid, q))

@info "Green's function" Rd=Rd R=R δ=δ grid_points=2^R

# --- 1. MI matrix (sampled; the Green's function is peaked on the dispersion,
#     so a decent sample budget is needed to resolve the structure) ---
n_samples = 200_000
@info "computing MI matrix" n_samples
@time W = MI.mutualinformation(qf, R; method=:uniform, n_samples)
@assert size(W) == (R, R)
@assert all(>=(0), W)

# --- 2. candidate bit layouts (as permutations of the grid-natural bits 1..R) ---
blocked_perm      = collect(1:R)                                  # w₁..w₆,kx₁..kx₆,ky₁..ky₆
interleaved_perm  = Int[r + off for r in 1:Rd for off in (0, Rd, 2Rd)]
# Found by `green_layout_search.jl`: keep coordinate blocks, but put w between
# the two momentum coordinates and reverse the two right-hand blocks. This is not
# a pairwise-MI MinLA optimum; it is a structure-aware Green's-function layout.
sandwich_perm     = [collect((Rd + 1):(2Rd)); collect(Rd:-1:1); collect((3Rd):-1:(2Rd + 1))]
@assert sort(interleaved_perm) == collect(1:R)
@assert sort(sandwich_perm) == collect(1:R)

Random.seed!(3)
@info "finding MI-optimal layout (MinLA on MI matrix)"
@time mi_perm, mi_cost, _ = MI.mi_ordering(W; max_deg=2, max_iter=200_000,
                                            initial_temp=1.0, final_temp=1e-4)

# baseline: a few random orderings
Random.seed!(4)
rand_perms = [shuffle(1:R) for _ in 1:4]

# --- 3. measure actual TT bond dimensions via TCI, for a given layout ---
localdims = fill(2, R)

function tci_bonddims(name::AbstractString, perm::Vector{Int}; tolerance=1e-8)
    invp = invperm(perm)
    # g(u) evaluates the Green's function at the grid-natural quantics index
    # obtained by un-permuting the chain index u. The resulting TT chain order
    # is `perm`, so its linkdims are the bond dimensions under that layout.
    g(u) = qf(u[invp])
    pivots = [ones(Int, R)]
    # a couple of random initial pivots improve robustness
    push!(pivots, rand(1:2, R))
    push!(pivots, rand(1:2, R))
    tci, ranks, errors = TCI.crossinterpolate2(ComplexF64, g, localdims, pivots;
        tolerance=tolerance, maxiter=100)
    χ = [TCI.linkdim(tci, k) for k in 1:R-1]
    final_err = isempty(errors) ? NaN : errors[end]
    @printf("  %-12s  χmax=%3d  χsum=%4d  χ=%s  (final err=%.2e)\n",
        name, maximum(χ), sum(χ), χ, final_err)
    return (name=name, perm=perm, χ=χ, χmax=maximum(χ), χsum=sum(χ),
            minla=MI.minla_cost(W, perm))
end

println("\n=== TT bond dimensions (tolerance=1e-8) for each layout ===")
res_blocked     = tci_bonddims("blocked", blocked_perm)
res_interleaved = tci_bonddims("interleaved", interleaved_perm)
res_sandwich    = tci_bonddims("sandwich", sandwich_perm)
res_mi          = tci_bonddims("MI-optimal", mi_perm)
println("  ---- random baselines ----")
res_rand = [tci_bonddims("random_$i", p) for (i, p) in enumerate(rand_perms)]

# --- 4. summary ---
println("\n=== Summary ===")
@printf("  blocked      : χmax=%3d χsum=%4d  minla=%.2f\n", res_blocked.χmax, res_blocked.χsum, res_blocked.minla)
@printf("  interleaved  : χmax=%3d χsum=%4d  minla=%.2f\n", res_interleaved.χmax, res_interleaved.χsum, res_interleaved.minla)
@printf("  sandwich    : χmax=%3d χsum=%4d  minla=%.2f  perm=%s\n", res_sandwich.χmax, res_sandwich.χsum, res_sandwich.minla, res_sandwich.perm)
@printf("  MI-optimal   : χmax=%3d χsum=%4d  minla=%.2f  perm=%s\n", res_mi.χmax, res_mi.χsum, res_mi.minla, res_mi.perm)
χmax_rand = [r.χmax for r in res_rand]
@printf("  random       : χmax ∈ [%d, %d] (mean %.0f)\n", minimum(χmax_rand), maximum(χmax_rand), mean(χmax_rand))

# reduction vs blocked
χmax_b = res_blocked.χmax
for r in (res_interleaved, res_sandwich, res_mi)
    @printf("  %-12s vs blocked: χmax %.0f → %.0f (%.0f%% reduction), χsum %.0f → %.0f\n",
        r.name, χmax_b, r.χmax, 100 * (χmax_b - r.χmax) / χmax_b,
        res_blocked.χsum, r.χsum)
end

# save artifacts
using DelimitedFiles
writedlm(joinpath(@__DIR__, "out_2d_MI_matrix.csv"), W, ',')
open(joinpath(@__DIR__, "out_2d_summary.txt"), "w") do io
    println(io, "Rd=$Rd R=$R δ=$δ n_samples=$n_samples")
    println(io, "blocked     χmax=", res_blocked.χmax, " χsum=", res_blocked.χsum, " minla=", res_blocked.minla)
    println(io, "interleaved χmax=", res_interleaved.χmax, " χsum=", res_interleaved.χsum, " minla=", res_interleaved.minla)
    println(io, "sandwich   χmax=", res_sandwich.χmax, " χsum=", res_sandwich.χsum, " minla=", res_sandwich.minla, " perm=", res_sandwich.perm)
    println(io, "MI-optimal  χmax=", res_mi.χmax, " χsum=", res_mi.χsum, " minla=", res_mi.minla, " perm=", res_mi.perm)
    println(io, "random      χmax=", χmax_rand)
end
@info "wrote out_2d_MI_matrix.csv, out_2d_summary.txt"
