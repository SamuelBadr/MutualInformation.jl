using Pkg; Pkg.activate(@__DIR__)
using LinearAlgebra, Printf, Statistics
import QuanticsGrids as QG
import MutualInformation as MI

# Data-driven multi-resolution optimizer for the Green's function.
# For each Rd, generate a broad structured grammar automatically and score every
# candidate by exact SVD bond dimensions of the full quantics tensor. This is
# feasible at Rd<=6 (R<=18) and reveals stable layout families to lift to larger R.

δ = 0.2

function make_problem(Rd)
    R = 3Rd
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
    return spec, qf
end

function evaluate_all_exact(Rd)
    spec, qf = make_problem(Rd)
    R = spec.R
    println("\n=== Rd=$Rd R=$R exact tensor screening ===")
    @time ψ = MI.amplitude_tensor(qf, fill(2, R))
    ψ ./= norm(ψ)
    ev = MI.ExactTensorLayoutEvaluator(ψ; rtol=1e-10)
    cands = MI.generate_layout_candidates(spec)
    # Add explicit baselines with stable names.
    append!(cands, [
        MI.LayoutCandidate("baseline_blocked", collect(1:R), :baseline),
        MI.LayoutCandidate("baseline_interleaved", [r + off for r in 1:Rd for off in (0, Rd, 2Rd)], :baseline),
    ])
    # Deduplicate.
    seen = Set{Tuple{Vararg{Int}}}(); uniq = MI.LayoutCandidate[]
    for c in cands
        k = Tuple(c.perm)
        if !(k in seen)
            push!(uniq, c); push!(seen, k)
        end
    end
    cands = uniq
    results = [MI.evaluate_layout(ev, c) for c in cands]
    sort!(results, by=x -> (x.χmax, x.χsum, x.Ssum))

    println("Top layouts by exact χmax:")
    for r in results[1:min(12, end)]
        c = r.candidate
        @printf("%-38s %-14s χmax=%3d χsum=%4d Ssum=%7.3f perm=%s\n",
            c.name[1:min(end,38)], string(c.family), r.χmax, r.χsum, r.Ssum, c.perm)
    end
    blocked = only(filter(r -> r.candidate.perm == collect(1:R), results))
    best = first(results)
    @printf("Best improvement vs blocked: χmax %d → %d (%.1f%%), χsum %d → %d\n",
        blocked.χmax, best.χmax, 100 * (blocked.χmax - best.χmax) / blocked.χmax,
        blocked.χsum, best.χsum)
    return results
end

all_results = Dict{Int,Any}()
for Rd in 3:6
    all_results[Rd] = evaluate_all_exact(Rd)
end

println("\n=== Winning family by resolution ===")
for Rd in sort(collect(keys(all_results)))
    r = first(all_results[Rd])
    @printf("Rd=%d best=%-38s family=%-12s χmax=%d χsum=%d perm=%s\n",
        Rd, r.candidate.name[1:min(end,38)], string(r.candidate.family), r.χmax, r.χsum, r.candidate.perm)
end
