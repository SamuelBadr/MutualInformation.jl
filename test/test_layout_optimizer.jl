using Test
using LinearAlgebra
import MutualInformation as MI

@testset "Structured layout optimizer" begin
    @testset "BitLayoutSpec validation" begin
        spec = MI.BitLayoutSpec(Dict(:w => 1:3, :kx => 4:6, :ky => 7:9); group_order=[:w, :kx, :ky])
        @test spec.R == 9
        @test spec.group_order == [:w, :kx, :ky]
        @test_throws ArgumentError MI.BitLayoutSpec(Dict(:a => [1, 2], :b => [2, 3]))
        @test_throws ArgumentError MI.BitLayoutSpec(Dict(:a => [1, 3], :b => [4, 5]))
        @test_throws ArgumentError MI.BitLayoutSpec(Dict(:a => [1], :b => [2]); group_order=[:a])
    end

    @testset "candidate families include expected layouts" begin
        spec = MI.BitLayoutSpec(Dict(:w => 1:2, :kx => 3:4, :ky => 5:6); group_order=[:w, :kx, :ky])
        cands = MI.generate_layout_candidates(spec)
        perms = Set(Tuple(c.perm) for c in cands)
        @test Tuple([1, 2, 3, 4, 5, 6]) in perms                    # blocked
        @test Tuple([1, 3, 5, 2, 4, 6]) in perms                    # interleaved
        @test Tuple([3, 4, 2, 1, 6, 5]) in perms                    # Green-like sandwich
        @test Tuple([1, 3, 2, 4, 5, 6]) in perms                    # pair interleave w/kx then ky
        @test all(sort(c.perm) == 1:6 for c in cands)
        @test length(perms) == length(cands)
    end

    @testset "cutwidth objective" begin
        W = [0.0 2.0 0.0; 2.0 0.0 1.0; 0.0 1.0 0.0]
        perm = [1, 2, 3]
        cw = MI.cut_weights(W, perm)
        @test cw ≈ [2.0, 1.0]
        obj = MI.cutwidth_objective(W, perm)
        @test obj.maxcut ≈ 2.0
        @test obj.sumcut ≈ MI.minla_cost(W, perm)  # MinLA equals sum over cut weights
    end

    @testset "cut-sketch scores and data-driven beam search" begin
        # Product state has zero cut entropy for every prefix when enumerated exactly.
        fprod(x) = 1.0
        sc = MI.sketch_cut_score(fprod, 4, [1, 2]; nrow=16, ncol=16)
        @test sc.rank == 1
        @test sc.entropy ≈ 0.0 atol = 1e-10

        # Bell-pair product state has lower max rank in paired ordering than natural.
        fpair(x) = ((x[1] == x[4]) && (x[2] == x[3])) ? 1.0 : 0.0
        cands = MI.beam_search_layout_by_cut_sketch(fpair, 4; beam_width=4, nrow=16, ncol=16)
        @test !isempty(cands)
        @test all(sort(c.perm) == 1:4 for c in cands)
        best = first(cands)
        ψ = MI.amplitude_tensor(fpair, [2, 2, 2, 2])
        ψ ./= norm(ψ)
        @test maximum(MI.bond_dimensions(ψ, best.perm)) <= 2
    end

    @testset "exact evaluator and search" begin
        # Long-range pair state: [1,4,2,3] lowers χmax vs natural.
        f(x) = ((x[1] == x[4]) && (x[2] == x[3])) ? 1.0 : 0.0
        ψ = MI.amplitude_tensor(f, [2, 2, 2, 2])
        ψ ./= norm(ψ)
        bad = MI.LayoutCandidate("bad", [1, 2, 3, 4], :manual)
        good = MI.LayoutCandidate("good", [1, 4, 2, 3], :manual)
        ev = MI.ExactTensorLayoutEvaluator(ψ)
        ebad = MI.evaluate_layout(ev, bad)
        egood = MI.evaluate_layout(ev, good)
        @test ebad.χmax == 4
        @test egood.χmax == 2

        results = MI.search_layouts([bad, good]; evaluator=ev, top_k=2)
        @test first(results).candidate.name == "good"
    end
end
