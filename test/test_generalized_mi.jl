using Test
using LinearAlgebra
import MutualInformation as MI

@testset "Generalized MI diagnostics" begin
    @testset "known entangled states" begin
        f_bell(x) = (x == [1, 1] || x == [2, 2]) ? 1 / sqrt(2) : 0.0
        ψ = MI.amplitude_tensor(f_bell, [2, 2])
        @test MI.subsystem_entropy(ψ, [1]) ≈ log(2) atol = 1e-10
        @test MI.subsystem_entropy(ψ, [1, 2]) ≈ 0.0 atol = 1e-10
        @test MI.block_mutual_information(ψ, [1], [2]) ≈ 2log(2) atol = 1e-10

        f_ghz(x) = (x == [1, 1, 1] || x == [2, 2, 2]) ? 1 / sqrt(2) : 0.0
        ψ3 = MI.amplitude_tensor(f_ghz, [2, 2, 2])
        @test MI.block_mutual_information(ψ3, [1], [2]) ≈ log(2) atol = 1e-10
        @test MI.conditional_mutual_information(ψ3, [1], [2], [3]) ≈ log(2) atol = 1e-10
        @test MI.interaction_information(ψ3, [1], [2], [3]) ≈ 0.0 atol = 1e-10
    end

    @testset "synergy identity on coherent XOR-like state" begin
        # Uniform coherent superposition over x,y,z with z = xor(x,y) in 1/2 encoding.
        # Note: these diagnostics are quantum block quantities of the pure state,
        # not classical measurement MI; pairwise quantum MI is nonzero here.
        function f_xor(x)
            xb = x[1] - 1; yb = x[2] - 1; zb = x[3] - 1
            return (xor(Bool(xb), Bool(yb)) == Bool(zb)) ? 1.0 : 0.0
        end
        ψ = MI.amplitude_tensor(f_xor, [2, 2, 2])
        ψ ./= norm(ψ)
        syn = MI.synergy_information(ψ, [3], [1], [2])
        expected = MI.block_mutual_information(ψ, [3], [1, 2]) -
                   MI.block_mutual_information(ψ, [3], [1]) -
                   MI.block_mutual_information(ψ, [3], [2])
        @test syn ≈ expected atol = 1e-10
        @test MI.block_mutual_information(ψ, [3], [1, 2]) > MI.block_mutual_information(ψ, [3], [1])

        # Classical measurement MI has the expected XOR behavior: pairwise zero,
        # joint positive, positive synergy.
        @test MI.classical_block_mutual_information(ψ, [3], [1]) ≈ 0.0 atol = 1e-10
        @test MI.classical_block_mutual_information(ψ, [3], [2]) ≈ 0.0 atol = 1e-10
        @test MI.classical_block_mutual_information(ψ, [3], [1, 2]) ≈ log(2) atol = 1e-10
        @test MI.classical_synergy_information(ψ, [3], [1], [2]) ≈ log(2) atol = 1e-10
    end

    @testset "validation and matrices" begin
        ψ = MI.amplitude_tensor(x -> 1.0, [2, 2, 2])
        @test_throws ArgumentError MI.subsystem_entropy(ψ, [1, 1])
        @test_throws ArgumentError MI.block_mutual_information(ψ, [1, 2], [2, 3])
        M = MI.block_mi_matrix(ψ, [[1], [2], [3]])
        @test size(M) == (3, 3)
        @test M ≈ zeros(3, 3) atol = 1e-10
        @test MI.total_correlation(ψ, [1], [2], [3]) ≈ 0.0 atol = 1e-10
    end
end
