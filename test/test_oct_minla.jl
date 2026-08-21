using Test
using Random
using LinearAlgebra
using Graphs
import MutualInformation as MI
const OCT = MI.OptimalCommunicationTree

function brute_minla(W)
    n = size(W, 1)
    perm = zeros(Int, n)
    used = falses(n)
    best_cost = Inf
    best_perm = collect(1:n)

    function rec!(pos)
        if pos > n
            perm[1] <= perm[end] || return nothing  # quotient reversal symmetry
            c = OCT.minla_cost(W, perm)
            if c < best_cost
                best_cost = c
                best_perm = copy(perm)
            end
            return nothing
        end
        for v in 1:n
            if !used[v]
                perm[pos] = v
                used[v] = true
                rec!(pos + 1)
                used[v] = false
            end
        end
        return nothing
    end

    rec!(1)
    return best_perm, best_cost
end

@testset "OCT/MinLA path solver" begin
    @testset "input validation" begin
        @test_throws ArgumentError OCT.minla_cost([0.0 1.0 2.0], [1])
        @test_throws ArgumentError OCT.minla_cost([0.0 1.0; 2.0 0.0], [1, 2])
        @test_throws ArgumentError OCT.minla_cost([0.0 -1.0; -1.0 0.0], [1, 2])
        @test_throws ArgumentError OCT.minla_cost([0.0 NaN; NaN 0.0], [1, 2])
        @test_throws ArgumentError OCT.minla_cost([0.0 1.0; 1.0 0.0], [1, 1])
        @test_throws ArgumentError OCT.path_graph_from_ordering([1, 1])
    end

    @testset "path round trips and OCT equivalence" begin
        for perm in ([1], [1, 2], [3, 1, 4, 2])
            tree = OCT.path_graph_from_ordering(perm)
            recovered = OCT.ordering_from_path(tree)
            @test recovered == perm || recovered == reverse(perm)
        end

        W = [0.0 2.0 1.0 0.5;
             2.0 0.0 3.0 0.2;
             1.0 3.0 0.0 4.0;
             0.5 0.2 4.0 0.0]
        perm = [1, 2, 3, 4]
        tree = OCT.path_graph_from_ordering(perm)
        @test OCT.oct_cost(tree, W) ≈ OCT.minla_cost(W, perm)

        # Non-path connected tree must be rejected by ordering_from_path.
        star = SimpleGraph(4)
        add_edge!(star, 1, 2); add_edge!(star, 1, 3); add_edge!(star, 1, 4)
        @test_throws ArgumentError OCT.ordering_from_path(star)
    end

    @testset "exact_minla equals brute force" begin
        rng = MersenneTwister(1234)
        for n in 2:7
            for _ in 1:4
                A = rand(rng, n, n)
                W = (A + A') / 2
                W[diagind(W)] .= 0.0
                p_exact, c_exact = OCT.exact_minla(W)
                p_brute, c_brute = brute_minla(W)
                @test c_exact ≈ c_brute atol = 1e-10
                @test sort(p_exact) == collect(1:n)
                @test OCT.minla_cost(W, p_exact) ≈ c_exact atol = 1e-10
                @test OCT.minla_cost(W, p_brute) ≈ c_brute atol = 1e-10
            end
        end
    end

    @testset "solve_minla and solve_oct API" begin
        # Scattered clusters: exact optimum groups each cluster contiguously.
        W = zeros(Float64, 9, 9)
        clusters = [[1, 5, 9], [2, 6, 7], [3, 4, 8]]
        for c in clusters, i in c, j in c
            i < j && (W[i, j] = W[j, i] = 1.0)
        end
        p, c = OCT.solve_minla(W; algorithm=:auto, exact_threshold=9)
        @test c < OCT.minla_cost(W, collect(1:9))
        @test sort(p) == 1:9

        tree, ctree = OCT.solve_oct(W; max_deg=2, algorithm=:auto, exact_threshold=9)
        @test ctree ≈ c atol = 1e-10
        @test maximum(degree(tree, v) for v in 1:9) <= 2
        @test OCT.oct_cost(tree, W) ≈ ctree atol = 1e-10
        @test OCT.minla_cost(W, OCT.ordering_from_path(tree)) ≈ ctree atol = 1e-10

        result = OCT.solve_oct_problem(W, 2; algorithm=:auto, exact_threshold=9)
        @test result.cost ≈ c atol = 1e-10
        @test length(result.edges) == 8

        @test_throws ArgumentError OCT.solve_oct(W; max_deg=3)
        @test_throws ArgumentError OCT.solve_oct_problem(W, 3)
    end

    @testset "heuristic is valid and no worse than deterministic initializers" begin
        rng = MersenneTwister(42)
        n = 14
        A = rand(rng, n, n)
        W = (A + A') / 2
        W[diagind(W)] .= 0.0
        p, c = OCT.solve_minla(W; algorithm=:heuristic, rng=MersenneTwister(43), n_restarts=8)
        @test sort(p) == 1:n
        @test c ≈ OCT.minla_cost(W, p) atol = 1e-10
        @test c <= OCT.minla_cost(W, collect(1:n)) + 1e-10
    end
end
