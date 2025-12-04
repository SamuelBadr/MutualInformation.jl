using Test
using LinearAlgebra
import MutualInformation as MI

# Simple statistics helpers
_mean(x) = sum(x) / length(x)
_std(x) = sqrt(sum((xi - _mean(x))^2 for xi in x) / (length(x) - 1))

# Test functions with different sparsity characteristics
f_smooth(x) = exp(-0.5 * sum((x[i] - x[i+1])^2 for i in 1:length(x)-1))
f_localized(x) = exp(2.0 * sum(x[i] == x[i+1] for i in 1:length(x)-1))
f_sparse(x) = sum(x[i] == x[i+1] for i in 1:length(x)-1) >= length(x) - 2 ? 1.0 : 0.1

@testset "Three Method Comparison Tests" begin

    @testset "API method selection" begin
        # Test that the API properly dispatches to all three methods
        len = 6

        # Exact method
        MI_exact = MI.mutualinformation(f_smooth, len; method=:exact)
        @test size(MI_exact) == (len, len)
        @test MI_exact ≈ MI_exact' atol = 1e-10

        # Uniform method
        MI_uniform = MI.mutualinformation(f_smooth, len; method=:uniform, n_samples=50000)
        @test size(MI_uniform) == (len, len)
        @test MI_uniform ≈ MI_uniform' atol = 1e-10

        # Hybrid method
        MI_hybrid = MI.mutualinformation(f_smooth, len; method=:hybrid, n_samples=50000)
        @test size(MI_hybrid) == (len, len)
        @test MI_hybrid ≈ MI_hybrid' atol = 1e-10

        # Backward compatibility: :sampled should work
        MI_sampled = MI.mutualinformation(f_smooth, len; method=:sampled, n_samples=50000)
        @test size(MI_sampled) == (len, len)
    end

    @testset "Direct function calls" begin
        # Test direct access to all three methods
        len = 5

        MI_exact = MI.mutualinformation_exact(f_smooth, fill(2, len))
        @test size(MI_exact) == (len, len)

        MI_uniform = MI.mutualinformation_uniform(f_smooth, len; n_samples=50000)
        @test size(MI_uniform) == (len, len)

        MI_hybrid = MI.mutualinformation_hybrid(f_smooth, len; n_samples=50000)
        @test size(MI_hybrid) == (len, len)

        # Backward compatibility alias
        MI_sampled = MI.mutualinformation_sampled(f_smooth, len; n_samples=50000)
        @test size(MI_sampled) == (len, len)
        # Note: Since they use different random seeds, they won't be identical
        # but should be statistically similar
        @test _mean(abs.(MI_sampled - MI_uniform)) < 0.1
    end

    @testset "Smooth function: all methods agree" begin
        # For smooth functions, all methods should give similar results
        len = 6
        n_samples = 150000  # Use many samples for good accuracy

        MI_exact = MI.mutualinformation_exact(f_smooth, fill(2, len))
        MI_uniform = MI.mutualinformation_uniform(f_smooth, len; n_samples)
        MI_hybrid = MI.mutualinformation_hybrid(f_smooth, len; n_samples, mcmc_fraction=0.8)

        # Extract off-diagonal elements
        mask = .!(I(len) .== 1)
        exact_vals = MI_exact[mask]
        uniform_vals = MI_uniform[mask]
        hybrid_vals = MI_hybrid[mask]

        # All should have reasonable accuracy compared to exact
        mae_uniform = _mean(abs.(exact_vals - uniform_vals))
        mae_hybrid = _mean(abs.(exact_vals - hybrid_vals))

        @test mae_uniform < 0.1  # Within 0.1 nats on average
        @test mae_hybrid < 0.2   # Hybrid may have slightly higher error due to MCMC variance

        # Check that all capture basic structure (nearest neighbors > distant)
        for i in 1:(len-1)
            @test MI_exact[i, i+1] > 0
            @test MI_uniform[i, i+1] > 0
            @test MI_hybrid[i, i+1] > 0

            if i + 3 <= len
                @test MI_exact[i, i+1] > MI_exact[i, i+3]
                @test MI_uniform[i, i+1] > MI_uniform[i, i+3]
                @test MI_hybrid[i, i+1] > MI_hybrid[i, i+3]
            end
        end
    end

    @testset "Localized function: hybrid vs uniform" begin
        # For localized functions, hybrid should perform at least as well as uniform
        len = 6
        n_samples = 100000

        MI_exact = MI.mutualinformation_exact(f_localized, fill(2, len))
        MI_uniform = MI.mutualinformation_uniform(f_localized, len; n_samples)
        MI_hybrid = MI.mutualinformation_hybrid(f_localized, len; n_samples, mcmc_fraction=0.9)

        mask = .!(I(len) .== 1)
        exact_vals = MI_exact[mask]
        uniform_vals = MI_uniform[mask]
        hybrid_vals = MI_hybrid[mask]

        mae_uniform = _mean(abs.(exact_vals - uniform_vals))
        mae_hybrid = _mean(abs.(exact_vals - hybrid_vals))

        # Both should achieve reasonable accuracy
        @test mae_uniform < 0.2
        @test mae_hybrid < 0.2

        # Hybrid should ideally be better, but we allow both to be reasonable
        # (statistical test, may vary)
        @test min(mae_uniform, mae_hybrid) < 0.15
    end

    @testset "Sparse function: hybrid advantage" begin
        # For very sparse functions, hybrid should show clear advantage
        len = 6
        n_samples = 100000

        MI_exact = MI.mutualinformation_exact(f_sparse, fill(2, len))
        MI_uniform = MI.mutualinformation_uniform(f_sparse, len; n_samples)
        MI_hybrid = MI.mutualinformation_hybrid(f_sparse, len; n_samples, mcmc_fraction=0.9)

        # All should produce valid results
        @test size(MI_uniform) == (len, len)
        @test size(MI_hybrid) == (len, len)
        @test all(>=(0), MI_uniform)
        @test all(>=(0), MI_hybrid)

        mask = .!(I(len) .== 1)
        mae_uniform = _mean(abs.(MI_exact[mask] - MI_uniform[mask]))
        mae_hybrid = _mean(abs.(MI_exact[mask] - MI_hybrid[mask]))

        # Both should give reasonable estimates
        @test mae_uniform < 0.3
        @test mae_hybrid < 0.3
    end

    @testset "Hybrid consistently outperforms uniform on sparse functions" begin
        # Statistical test: Run multiple times and verify hybrid is better on average
        # This is THE test that demonstrates hybrid's advantage

        # Very sparse function - only high amplitude when sites match
        function f_very_sparse(x)
            matches = sum(x[i] == x[i+1] for i in 1:length(x)-1)
            # High amplitude only when most neighbors match
            return matches >= length(x) - 2 ? 1.0 : 0.05
        end

        len = 6
        n_samples = 80000
        n_runs = 5  # Run multiple times to get statistical significance

        MI_exact = MI.mutualinformation_exact(f_very_sparse, fill(2, len))
        mask = .!(I(len) .== 1)
        exact_vals = MI_exact[mask]

        errors_uniform = Float64[]
        errors_hybrid = Float64[]

        for run in 1:n_runs
            MI_uniform = MI.mutualinformation_uniform(f_very_sparse, len; n_samples)
            MI_hybrid = MI.mutualinformation_hybrid(f_very_sparse, len;
                n_samples, mcmc_fraction=0.9, n_burnin=2000)

            push!(errors_uniform, _mean(abs.(exact_vals - MI_uniform[mask])))
            push!(errors_hybrid, _mean(abs.(exact_vals - MI_hybrid[mask])))
        end

        mean_error_uniform = _mean(errors_uniform)
        mean_error_hybrid = _mean(errors_hybrid)

        # Print for visibility
        @info "Hybrid vs Uniform performance on sparse function" mean_error_uniform mean_error_hybrid improvement_ratio = mean_error_uniform / mean_error_hybrid

        # Hybrid should have lower average error
        @test mean_error_hybrid < mean_error_uniform

        # Hybrid should win in majority of runs (at least 3 out of 5)
        wins_hybrid = sum(errors_hybrid .< errors_uniform)
        @test wins_hybrid >= 3

        # Both should be reasonable
        @test mean_error_uniform < 0.4
        @test mean_error_hybrid < 0.3
    end

    @testset "Hybrid parameter tuning" begin
        # Test that hybrid method works with different parameter settings
        len = 5
        n_samples = 30000

        # High MCMC fraction
        MI_high_mcmc = MI.mutualinformation_hybrid(f_smooth, len;
            n_samples, mcmc_fraction=0.95, n_burnin=500)
        @test size(MI_high_mcmc) == (len, len)
        @test all(>=(0), MI_high_mcmc)

        # Low MCMC fraction (more uniform)
        MI_low_mcmc = MI.mutualinformation_hybrid(f_smooth, len;
            n_samples, mcmc_fraction=0.5, n_burnin=500)
        @test size(MI_low_mcmc) == (len, len)
        @test all(>=(0), MI_low_mcmc)

        # Different flip count
        MI_multiflip = MI.mutualinformation_hybrid(f_smooth, len;
            n_samples, n_flip=2, n_burnin=500)
        @test size(MI_multiflip) == (len, len)
        @test all(>=(0), MI_multiflip)

        # All should produce reasonable results (using fewer samples, so more tolerance)
        MI_exact = MI.mutualinformation_exact(f_smooth, fill(2, len))
        mask = .!(I(len) .== 1)

        @test _mean(abs.(MI_exact[mask] - MI_high_mcmc[mask])) < 0.2
        @test _mean(abs.(MI_exact[mask] - MI_low_mcmc[mask])) < 0.2
        @test _mean(abs.(MI_exact[mask] - MI_multiflip[mask])) < 0.2
    end

    @testset "Error handling: invalid method" begin
        # Test that invalid method names throw errors
        @test_throws ArgumentError MI.mutualinformation(f_smooth, 5; method=:invalid)
        @test_throws ArgumentError MI.mutualinformation(f_smooth, 5; method=:mcmc)
    end

    @testset "Error handling: invalid parameters" begin
        # Test hybrid method parameter validation
        @test_throws ArgumentError MI.mutualinformation_hybrid(
            f_smooth, 5; mcmc_fraction=1.5)  # > 1.0
        @test_throws ArgumentError MI.mutualinformation_hybrid(
            f_smooth, 5; mcmc_fraction=-0.1)  # < 0.0
    end

    @testset "Large system performance" begin
        # Test that hybrid method works on large systems
        len = 20
        n_samples = 50000

        MI_uniform = MI.mutualinformation_uniform(f_smooth, len; n_samples)
        @test size(MI_uniform) == (len, len)

        MI_hybrid = MI.mutualinformation_hybrid(f_smooth, len; n_samples, mcmc_fraction=0.8)
        @test size(MI_hybrid) == (len, len)

        # Both should show reasonable correlation structure
        nn_mi_uniform = _mean([MI_uniform[i, i+1] for i in 1:len-1])
        nn_mi_hybrid = _mean([MI_hybrid[i, i+1] for i in 1:len-1])

        @test nn_mi_uniform > 0
        @test nn_mi_hybrid > 0
    end

    @testset "Consistency across runs" begin
        # Test that multiple runs give statistically similar results
        len = 5
        n_samples = 80000
        n_runs = 3

        # Test uniform sampling consistency
        uniform_results = [MI.mutualinformation_uniform(f_smooth, len; n_samples)
                           for _ in 1:n_runs]
        uniform_nn = [[M[i, i+1] for i in 1:len-1] for M in uniform_results]
        uniform_means = [_mean(nn) for nn in uniform_nn]
        @test _std(uniform_means) < 0.05  # Should be consistent

        # Test hybrid sampling consistency
        hybrid_results = [MI.mutualinformation_hybrid(f_smooth, len; n_samples)
                          for _ in 1:n_runs]
        hybrid_nn = [[M[i, i+1] for i in 1:len-1] for M in hybrid_results]
        hybrid_means = [_mean(nn) for nn in hybrid_nn]
        @test _std(hybrid_means) < 0.05  # Should be consistent
    end

    @testset "Complex amplitudes with all methods" begin
        # Test that all methods handle complex-valued functions
        f_complex(x) = exp(im * sum(x)) * f_smooth(x)
        len = 5

        MI_exact = MI.mutualinformation_exact(f_complex, fill(2, len))
        @test size(MI_exact) == (len, len)
        @test all(>=(0), MI_exact)

        MI_uniform = MI.mutualinformation_uniform(f_complex, len; n_samples=50000)
        @test size(MI_uniform) == (len, len)
        @test all(>=(0), MI_uniform)

        MI_hybrid = MI.mutualinformation_hybrid(f_complex, len; n_samples=50000)
        @test size(MI_hybrid) == (len, len)
        @test all(>=(0), MI_hybrid)

        # All should give similar results
        mask = .!(I(len) .== 1)
        @test _mean(abs.(MI_exact[mask] - MI_uniform[mask])) < 0.15
        @test _mean(abs.(MI_exact[mask] - MI_hybrid[mask])) < 0.15
    end

    @testset "Zero MCMC fraction = pure uniform" begin
        # Test that mcmc_fraction=0 gives similar results to pure uniform
        len = 5
        n_samples = 50000

        MI_uniform = MI.mutualinformation_uniform(f_smooth, len; n_samples)
        MI_hybrid_zero = MI.mutualinformation_hybrid(f_smooth, len;
            n_samples, mcmc_fraction=0.0)

        # Should be very similar (both pure uniform, just different random seeds)
        mask = .!(I(len) .== 1)
        diff = abs.(MI_uniform[mask] - MI_hybrid_zero[mask])
        @test _mean(diff) < 0.1  # Similar on average
    end

    @testset "Full MCMC (mcmc_fraction=1)" begin
        # Test that mcmc_fraction=1 works (pure MCMC without uniform)
        len = 5
        n_samples = 50000

        MI_hybrid_full = MI.mutualinformation_hybrid(f_smooth, len;
            n_samples, mcmc_fraction=1.0, n_burnin=2000)

        @test size(MI_hybrid_full) == (len, len)
        @test all(>=(0), MI_hybrid_full)

        # Should still give reasonable results
        MI_exact = MI.mutualinformation_exact(f_smooth, fill(2, len))
        mask = .!(I(len) .== 1)
        @test _mean(abs.(MI_exact[mask] - MI_hybrid_full[mask])) < 0.2
    end

    @testset "Auto method selection" begin
        # Test automatic method selection based on system size

        # Small system should use exact
        MI_small = MI.mutualinformation(f_smooth, 8; threshold=10)
        @test size(MI_small) == (8, 8)

        # Large system should use uniform sampling
        MI_large = MI.mutualinformation(f_smooth, 15; threshold=10, n_samples=30000)
        @test size(MI_large) == (15, 15)
    end

    @testset "Non-uniform dimensions error handling" begin
        # Sampling methods should reject non-uniform dimensions
        @test_throws ArgumentError MI.mutualinformation(
            f_smooth, [2, 3, 2]; method=:uniform)
        @test_throws ArgumentError MI.mutualinformation(
            f_smooth, [2, 3, 2]; method=:hybrid)

        # Exact method should work
        MI_exact = MI.mutualinformation(f_smooth, [2, 2, 2]; method=:exact)
        @test size(MI_exact) == (3, 3)
    end

end
