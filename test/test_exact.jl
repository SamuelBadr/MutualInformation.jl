using Test
using LinearAlgebra
import QuanticsGrids as QG
import MutualInformation as MI

@testset "Exact Method Tests" begin

    @testset "Product state (no entanglement)" begin
        # For a product state |ψ⟩ = |0⟩⊗|0⟩, MI should be 0
        # Here f returns 1 only for [1,1] configuration, 0 otherwise
        f(x) = x == [1, 1] ? 1.0 : 0.0
        MI_matrix = MI.mutualinformation_exact(Float64, f, [2, 2])

        # Mutual information should be 0 for product states
        @test MI_matrix[1, 2] ≈ 0.0 atol=1e-10
        @test MI_matrix[2, 1] ≈ 0.0 atol=1e-10

        # Diagonal should be 0
        @test MI_matrix[1, 1] == 0.0
        @test MI_matrix[2, 2] == 0.0

        # Should be symmetric
        @test MI_matrix[1, 2] ≈ MI_matrix[2, 1] atol=1e-10
    end

    @testset "Maximally entangled Bell state" begin
        # Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2
        # Configurations: [1,1] -> 00, [1,2] -> 01, [2,1] -> 10, [2,2] -> 11
        f(x) = (x == [1, 1] || x == [2, 2]) ? 1.0/sqrt(2) : 0.0
        MI_matrix = MI.mutualinformation_exact(Float64, f, [2, 2])

        # For maximally entangled 2-qubit state, each qubit has entropy log(2)
        # and the joint entropy is 0, so MI = log(2) + log(2) - 0 = 2*log(2)
        expected_MI = 2 * log(2)
        @test MI_matrix[1, 2] ≈ expected_MI atol=1e-10
        @test MI_matrix[2, 1] ≈ expected_MI atol=1e-10

        # Should be symmetric
        @test MI_matrix[1, 2] ≈ MI_matrix[2, 1] atol=1e-10
    end

    @testset "Three-qubit GHZ state" begin
        # GHZ state |GHZ⟩ = (|000⟩ + |111⟩)/√2
        f(x) = (x == [1, 1, 1] || x == [2, 2, 2]) ? 1.0/sqrt(2) : 0.0
        MI_matrix = MI.mutualinformation_exact(Float64, f, [2, 2, 2])

        # For GHZ state, the reduced 2-qubit density matrix is a classical mixture:
        # ρ_AB = 1/2(|00⟩⟨00| + |11⟩⟨11|) with entropy log(2)
        # Each single qubit is maximally mixed with entropy log(2)
        # So MI = S(A) + S(B) - S(AB) = log(2) + log(2) - log(2) = log(2)
        expected_MI = log(2)

        @test MI_matrix[1, 2] ≈ expected_MI atol=1e-10
        @test MI_matrix[1, 3] ≈ expected_MI atol=1e-10
        @test MI_matrix[2, 3] ≈ expected_MI atol=1e-10

        # Test symmetry
        @test MI_matrix[1, 2] ≈ MI_matrix[2, 1] atol=1e-10
        @test MI_matrix[1, 3] ≈ MI_matrix[3, 1] atol=1e-10
        @test MI_matrix[2, 3] ≈ MI_matrix[3, 2] atol=1e-10

        # Diagonal should be 0
        @test all(diag(MI_matrix) .== 0.0)
    end

    @testset "W-state (symmetric entanglement)" begin
        # W-state: (|100⟩ + |010⟩ + |001⟩)/√3
        f(x) = (count(==(2), x) == 1) ? 1.0/sqrt(3) : 0.0
        MI_matrix = MI.mutualinformation_exact(Float64, f, [2, 2, 2])

        # W-state has non-zero pairwise MI
        # All pairs should have equal MI due to symmetry
        @test MI_matrix[1, 2] ≈ MI_matrix[1, 3] atol=1e-10
        @test MI_matrix[1, 2] ≈ MI_matrix[2, 3] atol=1e-10
        @test MI_matrix[1, 2] > 0.0

        # Test symmetry
        @test MI_matrix ≈ MI_matrix' atol=1e-10

        # Diagonal should be 0
        @test all(diag(MI_matrix) .== 0.0)
    end

    @testset "Mixed state (maximally mixed)" begin
        # Uniform mixture: equal probability for all configurations
        f(_) = 1.0 / sqrt(4)  # 2^2 configurations for 2 qubits
        MI_matrix = MI.mutualinformation_exact(Float64, f, [2, 2])

        # For maximally mixed state, MI should be 0
        @test MI_matrix[1, 2] ≈ 0.0 atol=1e-10
        @test MI_matrix[2, 1] ≈ 0.0 atol=1e-10
    end

    @testset "Non-uniform local dimensions" begin
        # Test with different local dimensions (qubit + qutrit)
        # Product state
        f(x) = x == [1, 1] ? 1.0 : 0.0
        MI_matrix = MI.mutualinformation_exact(Float64, f, [2, 3])

        @test size(MI_matrix) == (2, 2)
        @test MI_matrix[1, 2] ≈ 0.0 atol=1e-10
        @test MI_matrix[2, 1] ≈ 0.0 atol=1e-10
    end

    @testset "Symmetry and general properties" begin
        # General test for properties using a smooth state
        f(x) = exp(-sum((xi-1.5)^2 for xi in x))
        MI_matrix = MI.mutualinformation_exact(Float64, f, [2, 2, 2])

        # Test that matrix is square
        @test size(MI_matrix, 1) == size(MI_matrix, 2) == 3

        # Test symmetry: MI(A:B) = MI(B:A)
        @test MI_matrix ≈ MI_matrix' atol=1e-10

        # Test diagonal is zero: MI(A:A) = 0
        @test all(diag(MI_matrix) .== 0.0)

        # Test non-negativity: MI ≥ 0
        @test all(MI_matrix .>= -1e-10)  # Allow small numerical errors
    end

    @testset "Complex amplitudes" begin
        # Test with complex-valued wave function
        f(x) = x == [1, 1] ? (1.0 + 0.0im)/sqrt(2) :
               x == [2, 2] ? (0.0 + 1.0im)/sqrt(2) : (0.0 + 0.0im)
        MI_matrix = MI.mutualinformation_exact(ComplexF64, f, [2, 2])

        # Should still be maximally entangled
        expected_MI = 2 * log(2)
        @test MI_matrix[1, 2] ≈ expected_MI atol=1e-10
    end

    @testset "Single site (edge case)" begin
        # Edge case: single site (MI matrix should be 1x1 with value 0)
        f(x) = x == [1] ? 1.0 : 0.0
        MI_matrix = MI.mutualinformation_exact(Float64, f, [2])

        @test size(MI_matrix) == (1, 1)
        @test MI_matrix[1, 1] == 0.0
    end

    @testset "Four-qubit cluster state" begin
        # Cluster state with specific entanglement pattern
        # More complex entanglement structure
        f(x) = begin
            # Simplified cluster-like state
            if x == [1,1,1,1] || x == [2,2,2,2]
                return 1.0/sqrt(2)
            elseif x == [1,2,1,2] || x == [2,1,2,1]
                return 1.0/sqrt(8)
            else
                return 0.0
            end
        end

        MI_matrix = MI.mutualinformation_exact(Float64, f, [2, 2, 2, 2])

        # Basic properties
        @test size(MI_matrix) == (4, 4)
        @test MI_matrix ≈ MI_matrix' atol=1e-10
        @test all(diag(MI_matrix) .== 0.0)
        @test all(MI_matrix .>= -1e-10)
    end

    @testset "QuanticsGrids integration" begin
        # Test with actual QuanticsGrids function
        R = 3  # Use small R for fast testing
        grid = QG.DiscretizedGrid(R, -1.0, +1.0)
        f(x) = exp(-x^2)
        qf(qx) = f(QG.quantics_to_origcoord(grid, qx))

        MI_matrix = MI.mutualinformation_exact(Float64, qf, fill(2, R))

        # Basic checks
        @test size(MI_matrix) == (R, R)
        @test MI_matrix ≈ MI_matrix' atol=1e-10  # Symmetric
        @test all(diag(MI_matrix) .== 0.0)  # Diagonal zero
        @test all(MI_matrix .>= -1e-10)  # Non-negative

        # For smooth Gaussian, nearby bits should have higher MI
        # This is a loose check since the exact values depend on discretization
        @test MI_matrix[1, 2] >= MI_matrix[1, 3] - 1e-6  # Nearest ≥ next-nearest (approximately)
    end

    @testset "Partial trace helper functions" begin
        # Test index conversions
        @test MI.index_to_config(0, [2, 2]) == [1, 1]
        @test MI.index_to_config(1, [2, 2]) == [2, 1]
        @test MI.index_to_config(2, [2, 2]) == [1, 2]
        @test MI.index_to_config(3, [2, 2]) == [2, 2]

        @test MI.config_to_index([1, 1], [2, 2]) == 0
        @test MI.config_to_index([2, 1], [2, 2]) == 1
        @test MI.config_to_index([1, 2], [2, 2]) == 2
        @test MI.config_to_index([2, 2], [2, 2]) == 3
    end

    @testset "Von Neumann entropy" begin
        # Test entropy calculation for known states

        # Pure state (zero entropy)
        ρ_pure = [1.0 0.0; 0.0 0.0]
        @test MI.von_neumann_entropy(ρ_pure) ≈ 0.0 atol=1e-10

        # Maximally mixed state (maximum entropy for 2D system)
        ρ_mixed = [0.5 0.0; 0.0 0.5]
        @test MI.von_neumann_entropy(ρ_mixed) ≈ log(2) atol=1e-10

        # Partially mixed state
        ρ_partial = [0.75 0.0; 0.0 0.25]
        expected_entropy = -0.75*log(0.75) - 0.25*log(0.25)
        @test MI.von_neumann_entropy(ρ_partial) ≈ expected_entropy atol=1e-10
    end

    @testset "Unified API - automatic method selection" begin
        # Test that unified API selects exact method for small systems
        f(x) = exp(-sum((xi-1.5)^2 for xi in x))

        # Small system should use exact method (default threshold=14)
        MI_auto = MI.mutualinformation(Float64, f, 8)
        MI_explicit_exact = MI.mutualinformation_exact(Float64, f, fill(2, 8))
        @test MI_auto ≈ MI_explicit_exact atol=1e-10

        # Can force exact method
        MI_forced = MI.mutualinformation(Float64, f, 10; method=:exact)
        @test size(MI_forced) == (10, 10)

        # Can adjust threshold
        MI_threshold = MI.mutualinformation(Float64, f, 12; threshold=12)
        MI_threshold_exact = MI.mutualinformation_exact(Float64, f, fill(2, 12))
        @test MI_threshold ≈ MI_threshold_exact atol=1e-10
    end

end
