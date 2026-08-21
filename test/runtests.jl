using Test
using LinearAlgebra
import QuanticsGrids as QG
import MutualInformation as MI

println("="^70)
println("Running MutualInformation.jl Test Suite")
println("="^70)

@testset "MutualInformation.jl" begin

    @testset "Basic QuanticsGrids integration" begin
        R = 5
        grid = QG.DiscretizedGrid(R, -1.0, +1.0)
        f(x) = exp(-x^2)
        qf(qx) = f(QG.quantics_to_origcoord(grid, qx))
        @test qf(fill(2, R)) == f(1 - 2 / 2^R)
    end

    # Include exact method tests
    println("\n" * "="^70)
    println("Testing Exact Method")
    println("="^70)
    include("test_exact.jl")

    # Include sampling method tests
    println("\n" * "="^70)
    println("Testing Sampling Method")
    println("="^70)
    include("test_sampling.jl")

    # Include OCT/MinLA path solver tests
    println("\n" * "="^70)
    println("Testing OCT/MinLA Path Solver")
    println("="^70)
    include("test_oct_minla.jl")

    # Include quantics layout tests
    println("\n" * "="^70)
    println("Testing Quantics Layout")
    println("="^70)
    include("test_quantics_layout.jl")

    # Include generalized MI tests
    println("\n" * "="^70)
    println("Testing Generalized MI")
    println("="^70)
    include("test_generalized_mi.jl")

    # Include structured layout optimizer tests
    println("\n" * "="^70)
    println("Testing Structured Layout Optimizer")
    println("="^70)
    include("test_layout_optimizer.jl")

    # # Include three-method comparison tests
    # println("\n" * "="^70)
    # println("Testing All Three Methods (Exact vs Uniform vs Hybrid)")
    # println("="^70)
    # include("test_three_methods.jl")

end

println("\n" * "="^70)
println("All tests completed!")
println("="^70)
