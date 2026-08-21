using Test
using LinearAlgebra
import QuanticsGrids as QG
import MutualInformation as MI
using Graphs

@testset "Quantics layout tests" begin

    @testset "amplitude_tensor" begin
        # Bell state on 2 sites
        f_bell(x) = (x == [1, 1] || x == [2, 2]) ? 1.0 / sqrt(2) : 0.0
        ψ = MI.amplitude_tensor(f_bell, [2, 2])
        @test size(ψ) == (2, 2)
        @test ψ[1, 1] ≈ 1 / sqrt(2)
        @test ψ[2, 2] ≈ 1 / sqrt(2)
        @test ψ[1, 2] ≈ 0.0
        @test ψ[2, 1] ≈ 0.0

        # GHZ3
        f_ghz(x) = (x == [1, 1, 1] || x == [2, 2, 2]) ? 1.0 / sqrt(2) : 0.0
        ψ3 = MI.amplitude_tensor(f_ghz, [2, 2, 2])
        @test size(ψ3) == (2, 2, 2)
        @test ψ3[1, 1, 1] ≈ 1 / sqrt(2)
        @test ψ3[2, 2, 2] ≈ 1 / sqrt(2)

        # complex amplitudes are promoted
        f_c(x) = (x == [1, 1]) ? (1.0 + 0.0im) : (0.0 + 0.0im)
        ψc = MI.amplitude_tensor(f_c, [2, 2])
        @test eltype(ψc) <: Complex

        # non-uniform local dims
        ψ_nu = MI.amplitude_tensor(x -> 1.0, [2, 3])
        @test size(ψ_nu) == (2, 3)
    end

    @testset "bond_dimensions: known states" begin
        # Bell state: rank-2 across the single cut
        f_bell(x) = (x == [1, 1] || x == [2, 2]) ? 1.0 / sqrt(2) : 0.0
        ψ = MI.amplitude_tensor(f_bell, [2, 2])
        @test MI.bond_dimensions(ψ) == [2]

        # Uniform superposition (product state): every unfolding is rank 1
        ψ_uniform = MI.amplitude_tensor(x -> 1.0, [2, 2, 2, 2])
        @test MI.bond_dimensions(ψ_uniform) == [1, 1, 1]

        # GHZ3: every cut rank 2
        f_ghz(x) = (x == [1, 1, 1] || x == [2, 2, 2]) ? 1.0 / sqrt(2) : 0.0
        ψ3 = MI.amplitude_tensor(f_ghz, [2, 2, 2])
        @test MI.bond_dimensions(ψ3) == [2, 2]
    end

    @testset "cut_entropies: known states" begin
        # Bell state: single cut entropy = log(2)
        f_bell(x) = (x == [1, 1] || x == [2, 2]) ? 1.0 / sqrt(2) : 0.0
        ψ = MI.amplitude_tensor(f_bell, [2, 2])
        @test MI.cut_entropies(ψ) ≈ [log(2)] atol = 1e-10

        # Product (uniform) state: zero entanglement everywhere
        ψ_uniform = MI.amplitude_tensor(x -> 1.0, [2, 2, 2])
        @test all(iszero, MI.cut_entropies(ψ_uniform))

        # GHZ3: log(2) on each cut
        f_ghz(x) = (x == [1, 1, 1] || x == [2, 2, 2]) ? 1.0 / sqrt(2) : 0.0
        ψ3 = MI.amplitude_tensor(f_ghz, [2, 2, 2])
        @test MI.cut_entropies(ψ3) ≈ [log(2), log(2)] atol = 1e-10
    end

    @testset "reordering reduces bond dimensions" begin
        # State with long-range entanglement: site 1 entangled with site 4,
        # site 2 entangled with site 3.
        f(x) = ((x[1] == x[4]) && (x[2] == x[3])) ? 1.0 : 0.0
        ψ = MI.amplitude_tensor(f, [2, 2, 2, 2])
        # sanity: normalise-independence of bond dims
        ψn = ψ ./ norm(ψ)

        # Natural ordering [1,2,3,4]: long-range pair (1,4) straddles the middle
        χ_nat = MI.bond_dimensions(ψn)
        @test χ_nat == [2, 4, 2]   # derived by hand

        # Reordered so each entangled pair is adjacent: [1,4,2,3]
        χ_re = MI.bond_dimensions(ψn, [1, 4, 2, 3])
        @test χ_re == [2, 1, 2]
        @test maximum(χ_re) < maximum(χ_nat)

        # Entropies follow the same pattern
        S_nat = MI.cut_entropies(ψn)
        S_re = MI.cut_entropies(ψn, [1, 4, 2, 3])
        @test maximum(S_re) < maximum(S_nat)
    end

    @testset "path_to_ordering" begin
        # A hand-built path 3-1-2-4 on 4 nodes (edges (3,1),(1,2),(2,4))
        g = SimpleGraph(4)
        add_edge!(g, 3, 1); add_edge!(g, 1, 2); add_edge!(g, 2, 4)
        order = MI.path_to_ordering(g)
        @test sort(order) == 1:4
        @test length(order) == 4
        # Walking from an endpoint must reproduce the path (up to reversal)
        @test order == [3, 1, 2, 4] || order == [4, 2, 1, 3]

        # Single node
        @test MI.path_to_ordering(SimpleGraph(1)) == [1]
    end

    @testset "minla_cost" begin
        # Symmetric and matches direct computation
        W = [0.0 2.0 0.0; 2.0 0.0 1.0; 0.0 1.0 0.0]
        @test MI.minla_cost(W, [1, 2, 3]) ≈ 2 * 1 + 0 * 2 + 1 * 1   # = 3.0
        @test MI.minla_cost(W, [3, 2, 1]) ≈ MI.minla_cost(W, [1, 2, 3])  # reversal-invariant
        # Identity permutation when L==1
        @test MI.minla_cost(zeros(1, 1), [1]) == 0.0
    end

    @testset "mi_ordering: clustered MI matrix" begin
        # Three clusters of mutually-informative sites, but scattered across the
        # label space so the natural ordering [1..9] is genuinely bad.
        W = zeros(Float64, 9, 9)
        clusters = [[1, 5, 9], [2, 6, 7], [3, 4, 8]]
        for c in clusters, i in c, j in c
            if i < j
                W[i, j] = W[j, i] = 1.0
            end
        end
        W[3, 4] = W[4, 3] = 0.05
        W[6, 7] = W[7, 6] = 0.05

        natural_cost = MI.minla_cost(W, collect(1:9))
        perm, cost, tree = MI.mi_ordering(W; max_iter=50000)
        @test sort(perm) == 1:9                # valid permutation
        @test maximum(degree(tree, v) for v in 1:9) <= 2   # path
        # MI-optimal ordering must improve on the scattered natural ordering
        @test cost < natural_cost
        # ... and the reported cost matches minla_cost of the returned permutation
        @test cost ≈ MI.minla_cost(W, perm) atol = 1e-9
    end

    @testset "end-to-end: MI-guided ordering on quantics function" begin
        # Small quantics function; verify the layout API is consistent with
        # the MI matrix and bond dimensions.
        R = 5
        grid = QG.DiscretizedGrid(R, -1.0, +1.0)
        f(x) = sin(4π * x) * exp(-x^2)
        qf(q) = f(QG.quantics_to_origcoord(grid, q))

        W = MI.mutualinformation(qf, R; method=:exact)
        ψ = MI.amplitude_tensor(qf, fill(2, R))

        perm, cost, _ = MI.mi_ordering(W; max_iter=20000)
        @test sort(perm) == 1:R

        # The reported cost equals minla_cost of the returned permutation
        @test cost ≈ MI.minla_cost(W, perm) atol = 1e-9

        # Bond dims computed two ways agree
        @test MI.bond_dimensions(ψ, perm) == MI.bond_dimensions(permutedims(ψ, perm))
    end
end
