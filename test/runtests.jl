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

end

println("\n" * "="^70)
println("All tests completed!")
println("="^70)
