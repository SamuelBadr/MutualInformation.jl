using Pkg; Pkg.activate(@__DIR__)
using LinearAlgebra, Statistics, Printf, Random
import QuanticsGrids as QG
import MutualInformation as MI

# -----------------------------------------------------------------------------
# Experiment 3: 2D function with PAIRWISE cross-coordinate correlations.
#   f(x, y) = exp(-((x - y)^2) / σ^2)
# is strongly correlated along the diagonal x = y, so quantics bit r of x is
# pairwise-correlated with quantics bit r of y. The natural blocked layout
# [x1..xR, y1..yR] places those paired bits far apart; the interleaved layout
# [x1, y1, x2, y2, ...] places them adjacent. The MI matrix should reveal the
# x_r <-> y_r pairs, so MI-guided MinLA should RECOVER the interleaved layout
# and beat blocked. This is the multidimensional case where pairwise MI is the
# RIGHT surrogate.
# -----------------------------------------------------------------------------

Rd = 6
R = 2Rd
x_inds = [[(:x, r)] for r in 1:Rd]
y_inds = [[(:y, r)] for r in 1:Rd]
indextable = [x_inds; y_inds]                       # blocked = grid-natural
grid = QG.DiscretizedGrid((:x, :y), indextable;
    lower_bound=(-1.0, -1.0), upper_bound=(1.0, 1.0))

σ = 0.3
f(x, y) = exp(-((x - y)^2) / σ^2)
qf(q) = f(QG.quantics_to_origcoord(grid, q)...)

@info "2D diagonal-correlated" Rd=Rd R=R σ=σ

# MI matrix: smooth, non-peaked function -> uniform sampling is accurate.
# (R=12 is small enough that we could go exact, but sampling is fast & sufficient
#  here and exercises the realistic large-R path.)
@time W = MI.mutualinformation(qf, R; method=:uniform, n_samples=300_000)
ψ = MI.amplitude_tensor(qf, fill(2, R)); ψ ./= norm(ψ)

blocked     = collect(1:R)
interleaved = Int[(r, r + Rd)[c] for r in 1:Rd for c in 1:2]   # x1,y1,x2,y2,...
@assert sort(interleaved) == collect(1:R)

Random.seed!(7)
@time mi_perm, mi_cost, _ = MI.mi_ordering(W; max_deg=2, max_iter=100_000,
    initial_temp=1.0, final_temp=1e-4)

# diagnostic: which pairs does the MI matrix rank highest?
function top_pairs(W, n=6)
    pairs = sort([(W[i, j], i, j) for i in 1:size(W,1) for j in (i+1):size(W,2)], rev=true)[1:n]
    return pairs
end
println("\ntop MI pairs (val, i, j)  [x bits 1..6, y bits 7..12]:")
for (v, i, j) in top_pairs(W)
    println("  ", round(v, digits=4), "  bit ", i, " <-> bit ", j)
end

function ev(perm)
    χ = MI.bond_dimensions(ψ, perm)
    S = MI.cut_entropies(ψ, perm)
    (perm=perm, minla=MI.minla_cost(W, perm), χmax=maximum(χ), χsum=sum(χ),
     smax=maximum(S), ssum=sum(S), χ=χ)
end
e_blk = ev(blocked); e_int = ev(interleaved); e_mi = ev(mi_perm)

# random baseline
Random.seed!(8)
e_rand = ev.(shuffle(1:R) for _ in 1:40)

println("\n=== Exact bond dimensions (R=$R) ===")
@printf("  blocked     χmax=%2d χsum=%3d Smax=%.4f Ssum=%.4f minla=%.3f\n", e_blk.χmax, e_blk.χsum, e_blk.smax, e_blk.ssum, e_blk.minla)
@printf("  interleaved χmax=%2d χsum=%3d Smax=%.4f Ssum=%.4f minla=%.3f\n", e_int.χmax, e_int.χsum, e_int.smax, e_int.ssum, e_int.minla)
@printf("  MI-optimal  χmax=%2d χsum=%3d Smax=%.4f Ssum=%.4f minla=%.3f\n", e_mi.χmax, e_mi.χsum, e_mi.smax, e_mi.ssum, e_mi.minla)
println("  MI-optimal perm: ", e_mi.perm, "  (interleaved would be ", interleaved, ")")
@printf("  random      χmax∈[%d,%d] mean=%.1f  minla∈[%.2f,%.2f]\n",
    minimum(e.χmax for e in e_rand), maximum(e.χmax for e in e_rand), mean(e.χmax for e in e_rand),
    minimum(e.minla for e in e_rand), maximum(e.minla for e in e_rand))

println("\n=== Reduction vs blocked ===")
for (name, e) in (("interleaved", e_int), ("MI-optimal", e_mi))
    @printf("  %-11s χmax %d → %d (%.0f%%), χsum %d → %d (%.0f%%), Ssum %.3f → %.3f\n",
        name, e_blk.χmax, e.χmax, 100 * (e_blk.χmax - e.χmax) / e_blk.χmax,
        e_blk.χsum, e.χsum, 100 * (e_blk.χsum - e.χsum) / e_blk.χsum,
        e_blk.ssum, e.ssum)
end

# did MI recover the interleaved layout (up to reversal)?
mi_set = Set([(min(e_mi.perm[k], e_mi.perm[k+1]), max(e_mi.perm[k], e_mi.perm[k+1])) for k in 1:R-1])
int_set = Set([(min(interleaved[k], interleaved[k+1]), max(interleaved[k], interleaved[k+1])) for k in 1:R-1])
println("\nMI-optimal shares ", length(mi_set ∩ int_set), "/", R - 1, " edges with the interleaved layout")

# surrogate correlation over all orderings
minla = [e_blk.minla, e_int.minla, e_mi.minla, [e.minla for e in e_rand]...]
χmax  = [e_blk.χmax,  e_int.χmax,  e_mi.χmax,  [e.χmax for e in e_rand]...]
ssum  = [e_blk.ssum,  e_int.ssum,  e_mi.ssum,  [e.ssum for e in e_rand]...]
println("\ncorr(MinLA, χmax) = ", round(cor(minla, χmax), digits=3),
        "   corr(MinLA, Ssum) = ", round(cor(minla, ssum), digits=3))

using DelimitedFiles
writedlm(joinpath(@__DIR__, "out_3d_MI_matrix.csv"), W, ',')
