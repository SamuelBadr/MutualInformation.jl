using Pkg; Pkg.activate(@__DIR__)
using LinearAlgebra, Statistics, Printf, Random
import QuanticsGrids as QG
import MutualInformation as MI

# -----------------------------------------------------------------------------
# Diagnostic: at small R we can compute EXACT MI and EXACT bond dimensions, so we
# can directly test whether corr(MinLA, bond_dim) is positive (surrogate works)
# or negative/zero (surrogate fails) for the Green's function. This separates a
# fundamental surrogate failure from a sampling artifact.
# -----------------------------------------------------------------------------

for Rd in (3, 4)            # R = 3*3=9 and 3*4=12
    R = 3Rd
    w_inds = [[(:w, r)]  for r in 1:Rd]
    x_inds = [[(:kx, r)] for r in 1:Rd]
    y_inds = [[(:ky, r)] for r in 1:Rd]
    indextable = [w_inds; x_inds; y_inds]
    grid = QG.DiscretizedGrid((:w, :kx, :ky), indextable;
        lower_bound=(-5.0, -1π, -1π), upper_bound=(+5.0, +1π, +1π))
    disp(k) = -2 * sum(cos, k)
    f(k; δ=0.2) = 1 / (k[1] - disp(k[2:end]) + im * 0.2)
    qf(q) = f(QG.quantics_to_origcoord(grid, q))

    W = MI.mutualinformation(qf, R; method=:exact)          # exact MI
    ψ = MI.amplitude_tensor(qf, fill(2, R)); ψ ./= norm(ψ)   # exact state

    blocked = collect(1:R)
    interleaved = Int[r + off for r in 1:Rd for off in (0, Rd, 2Rd)]
    Random.seed!(R)
    rperms = [shuffle(1:R) for _ in 1:120]

    function ev(perm)
        χ = MI.bond_dimensions(ψ, perm)
        (minla=MI.minla_cost(W, perm), χmax=maximum(χ), χsum=sum(χ),
         smax=maximum(MI.cut_entropies(ψ, perm)))
    end
    e_blk = ev(blocked); e_int = ev(interleaved)
    e_rand = ev.(rperms)
    e_mi, _ = MI.mi_ordering(W; max_deg=2, max_iter=80_000)
    e_mi = ev(e_mi)

    minla = [e_blk.minla, e_int.minla, e_mi.minla, [e.minla for e in e_rand]...]
    χmax = [e_blk.χmax, e_int.χmax, e_mi.χmax, [e.χmax for e in e_rand]...]
    χsum = [e_blk.χsum, e_int.χsum, e_mi.χsum, [e.χsum for e in e_rand]...]

    println("\n===== Rd=$Rd  R=$R  (exact MI, exact bond dims) =====")
    @printf("  blocked     χmax=%2d χsum=%3d minla=%.3f\n", e_blk.χmax, e_blk.χsum, e_blk.minla)
    @printf("  interleaved χmax=%2d χsum=%3d minla=%.3f\n", e_int.χmax, e_int.χsum, e_int.minla)
    @printf("  MI-optimal  χmax=%2d χsum=%3d minla=%.3f\n", e_mi.χmax, e_mi.χsum, e_mi.minla)
    @printf("  random      χmax∈[%d,%d] mean=%.1f  minla∈[%.2f,%.2f]\n",
        minimum(e.χmax for e in e_rand), maximum(e.χmax for e in e_rand), mean(e.χmax for e in e_rand),
        minimum(e.minla for e in e_rand), maximum(e.minla for e in e_rand))
    println("  correlations over all $(length(minla)) orderings:")
    @printf("    corr(MinLA, χmax) = %+.3f   (positive = surrogate works)\n", cor(minla, χmax))
    @printf("    corr(MinLA, χsum) = %+.3f\n", cor(minla, χsum))
    @printf("    corr(MinLA, χmax) among RANDOM only = %+.3f\n",
        cor([e.minla for e in e_rand], [e.χmax for e in e_rand]))
end
