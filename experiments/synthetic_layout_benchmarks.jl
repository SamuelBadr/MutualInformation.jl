using Pkg; Pkg.activate(@__DIR__)
using LinearAlgebra, Printf, Statistics
import QuanticsGrids as QG
import MutualInformation as MI

# Synthetic quantics-layout benchmark suite.
#
# Philosophy: generate functions by dependency motif, then evaluate layout
# candidates by the actual exact TT bond dimensions (SVD ranks of unfoldings).
# This creates controlled regression problems for pairwise, collider, chain, and
# star/higher-order structures relevant to Hubbard/Parquet objects.

struct SyntheticBenchmark
    name::String
    coords::Vector{Symbol}
    bounds::Vector{Tuple{Float64,Float64}}
    Rd::Int
    f::Function
    expected::String
end

function make_grid(coords, bounds, Rd)
    indextable = [[[(c, r)] for r in 1:Rd] for c in coords]
    indextable = reduce(vcat, indextable)
    lower = Tuple(first.(bounds)); upper = Tuple(last.(bounds))
    return QG.DiscretizedGrid(Tuple(coords), indextable; lower_bound=lower, upper_bound=upper)
end

function make_problem(b::SyntheticBenchmark)
    grid = make_grid(b.coords, b.bounds, b.Rd)
    qf(q) = b.f(QG.quantics_to_origcoord(grid, q)...)
    groups = Dict{Symbol,Vector{Int}}()
    for (i, c) in enumerate(b.coords)
        groups[c] = collect(((i - 1) * b.Rd + 1):(i * b.Rd))
    end
    spec = MI.BitLayoutSpec(groups; group_order=b.coords)
    return spec, qf
end

lorentz(t, δ=0.15) = 1 / (t + im * δ)
gauss(t, σ=0.25) = exp(-(t / σ)^2)

benchmarks = SyntheticBenchmark[
    SyntheticBenchmark(
        "separable_3coord",
        [:x, :y, :z],
        [(-1, 1), (-1, 1), (-1, 1)],
        5,
        (x, y, z) -> exp(-x^2) * exp(-2y^2) * exp(-0.5z^2),
        "Any block layout should be very low-rank; no strong inter-coordinate coupling."),

    SyntheticBenchmark(
        "pairwise_diagonal_xy",
        [:x, :y],
        [(-1, 1), (-1, 1)],
        7,
        (x, y) -> gauss(x - y, 0.22),
        "Pairwise same-scale x↔y; interleaving should win."),

    SyntheticBenchmark(
        "collider_sum_z_eq_x_plus_y",
        [:z, :x, :y],
        [(-2, 2), (-1, 1), (-1, 1)],
        6,
        (z, x, y) -> lorentz(z - x - y, 0.18),
        "Collider x,y→z; sandwich x-z-y or y-z-x should win."),

    SyntheticBenchmark(
        "chain_x_y_z",
        [:x, :y, :z],
        [(-1, 1), (-1, 1), (-1, 1)],
        6,
        (x, y, z) -> gauss(x - y, 0.28) * gauss(y - z, 0.28),
        "Chain x-y-z; y should be the middle block."),

    SyntheticBenchmark(
        "ph_transfer_q_eq_kp_minus_k",
        [:q, :k, :kp],
        [(-2, 2), (-1, 1), (-1, 1)],
        6,
        (q, k, kp) -> lorentz(q - (kp - k), 0.18),
        "Particle-hole transfer relation k,kp→q; q should be middle."),

    SyntheticBenchmark(
        "star4_s_eq_x_plus_y_plus_z",
        [:s, :x, :y, :z],
        [(-3, 3), (-1, 1), (-1, 1), (-1, 1)],
        4,
        (s, x, y, z) -> lorentz(s - x - y - z, 0.20),
        "Four-variable star/collider; s should be central with sources split around it."),
]

function candidate_baselines(spec::MI.BitLayoutSpec)
    Rd = length(first(values(spec.groups)))
    out = MI.LayoutCandidate[]
    push!(out, MI.LayoutCandidate("natural_blocked", collect(1:spec.R), :baseline))
    if allequal(length.(values(spec.groups)))
        push!(out, MI.LayoutCandidate("scale_interleaved_spec_order",
            [spec.groups[g][r] for r in 1:Rd for g in spec.group_order], :baseline))
    end
    return out
end

function dedup(cands)
    seen = Set{Tuple{Vararg{Int}}}(); out = MI.LayoutCandidate[]
    for c in cands
        k = Tuple(c.perm)
        if !(k in seen)
            push!(out, c); push!(seen, k)
        end
    end
    return out
end

function block_order_string(spec, perm)
    # If perm is exactly concatenated full groups (possibly reversed), summarize it.
    pos = 1
    parts = String[]
    while pos <= length(perm)
        matched = false
        for g in spec.group_order
            bits = spec.groups[g]
            for (suffix, seq) in (("", bits), ("ʳ", reverse(bits)))
                L = length(seq)
                if pos + L - 1 <= length(perm) && perm[pos:pos+L-1] == seq
                    push!(parts, string(g) * suffix)
                    pos += L
                    matched = true
                    break
                end
            end
            matched && break
        end
        matched || return "nonblock"
    end
    return join(parts, "-")
end

function summarize_by_family(results)
    fams = unique(r.candidate.family for r in results)
    for fam in sort(collect(fams); by=string)
        rf = filter(r -> r.candidate.family == fam, results)
        isempty(rf) && continue
        best = first(sort(rf, by=x -> (x.χmax, x.χsum, x.Ssum)))
        @printf("  best %-15s χmax=%4d χsum=%5d Ssum=%8.3f name=%s\n",
            string(fam), best.χmax, best.χsum, best.Ssum, best.candidate.name)
    end
end

function diagnostics(spec, ψ)
    groups = [spec.groups[g] for g in spec.group_order]
    println("  Classical block MI:")
    display(round.(MI.classical_block_mi_matrix(ψ, groups), digits=3))
    if length(groups) == 3
        for (i, g) in enumerate(spec.group_order)
            T = groups[i]
            srcs = [groups[j] for j in 1:3 if j != i]
            @printf("    target %-3s C I(T:others)=%.3f C synergy=%.3f CMI(others|T)=%.3f\n",
                string(g),
                MI.classical_block_mutual_information(ψ, T, reduce(vcat, srcs)),
                MI.classical_synergy_information(ψ, T, srcs...),
                MI.classical_conditional_mutual_information(ψ, srcs[1], srcs[2], T))
        end
    end
end

function run_benchmark(b::SyntheticBenchmark)
    println("\n", "="^88)
    println(b.name, "  Rd=", b.Rd, "  coords=", b.coords)
    println("Expected motif: ", b.expected)
    spec, qf = make_problem(b)
    @time ψ = MI.amplitude_tensor(qf, fill(2, spec.R))
    ψ ./= norm(ψ)

    diagnostics(spec, ψ)

    cands = dedup([MI.generate_layout_candidates(spec); candidate_baselines(spec)])
    ev = MI.ExactTensorLayoutEvaluator(ψ; rtol=1e-10)
    results = [MI.evaluate_layout(ev, c) for c in cands]
    sort!(results, by=x -> (x.χmax, x.χsum, x.Ssum))

    natural = only(filter(r -> r.candidate.perm == collect(1:spec.R), results))
    best = first(results)
    println("\nBest candidates:")
    for r in results[1:min(10, end)]
        c = r.candidate
        @printf("  %-36s %-15s χmax=%4d χsum=%5d Ssum=%8.3f block=%s perm=%s\n",
            c.name[1:min(end,36)], string(c.family), r.χmax, r.χsum, r.Ssum,
            block_order_string(spec, c.perm), c.perm)
    end
    println("\nBest by family:")
    summarize_by_family(results)
    @printf("\nNatural blocked: χmax=%d χsum=%d Ssum=%.3f\n", natural.χmax, natural.χsum, natural.Ssum)
    @printf("Best found     : χmax=%d χsum=%d Ssum=%.3f (%s, block=%s)\n",
        best.χmax, best.χsum, best.Ssum, best.candidate.name, block_order_string(spec, best.candidate.perm))
    @printf("Improvement    : χmax %.1f%%, χsum %.1f%%\n",
        100 * (natural.χmax - best.χmax) / max(natural.χmax, 1),
        100 * (natural.χsum - best.χsum) / max(natural.χsum, 1))
    return (benchmark=b, spec=spec, results=results)
end

all_results = [run_benchmark(b) for b in benchmarks]

println("\n", "="^88)
println("Summary")
for item in all_results
    b = item.benchmark; spec = item.spec; results = item.results
    nat = only(filter(r -> r.candidate.perm == collect(1:spec.R), results))
    best = first(results)
    @printf("%-32s best χmax=%4d vs natural=%4d  best=%-30s block=%s\n",
        b.name, best.χmax, nat.χmax, best.candidate.name[1:min(end,30)], block_order_string(spec, best.candidate.perm))
end
