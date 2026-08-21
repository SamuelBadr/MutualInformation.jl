using Pkg; Pkg.activate(@__DIR__)
using LinearAlgebra, Printf
import QuanticsGrids as QG
import MutualInformation as MI

function green_problem(Rd; δ=0.2)
    R = 3Rd
    groups = Dict(:w => collect(1:Rd), :kx => collect((Rd+1):(2Rd)), :ky => collect((2Rd+1):(3Rd)))
    w_inds = [[(:w, r)] for r in 1:Rd]
    x_inds = [[(:kx, r)] for r in 1:Rd]
    y_inds = [[(:ky, r)] for r in 1:Rd]
    grid = QG.DiscretizedGrid((:w, :kx, :ky), [w_inds; x_inds; y_inds];
        lower_bound=(-5.0, -1π, -1π), upper_bound=(5.0, 1π, 1π))
    disp(k) = -2 * sum(cos, k)
    f(k; δ=δ) = 1 / (k[1] - disp(k[2:end]) + im * δ)
    qf(q) = f(QG.quantics_to_origcoord(grid, q))
    return groups, qf
end

function diagonal_problem(Rd; σ=0.3)
    R = 2Rd
    groups = Dict(:x => collect(1:Rd), :y => collect((Rd+1):(2Rd)))
    x_inds = [[(:x, r)] for r in 1:Rd]
    y_inds = [[(:y, r)] for r in 1:Rd]
    grid = QG.DiscretizedGrid((:x, :y), [x_inds; y_inds]; lower_bound=(-1.0, -1.0), upper_bound=(1.0, 1.0))
    f(x, y) = exp(-((x - y)^2) / σ^2)
    qf(q) = f(QG.quantics_to_origcoord(grid, q)...)
    return groups, qf
end

function print_green_diag(Rd)
    groups, qf = green_problem(Rd)
    R = 3Rd
    ψ = MI.amplitude_tensor(qf, fill(2, R)); ψ ./= norm(ψ)
    W,KX,KY = groups[:w], groups[:kx], groups[:ky]
    println("\n=== Green generalized MI Rd=$Rd R=$R ===")
    println("Quantum block MI matrix [w,kx,ky]:")
    display(round.(MI.block_mi_matrix(ψ, [W,KX,KY]), digits=4))
    println("Classical block MI matrix [w,kx,ky]:")
    display(round.(MI.classical_block_mi_matrix(ψ, [W,KX,KY]), digits=4))
    for target in (:w, :kx, :ky)
        T = groups[target]
        srcs = [groups[g] for g in (:w, :kx, :ky) if g != target]
        jointq = MI.block_mutual_information(ψ, T, reduce(vcat, srcs))
        synq = MI.synergy_information(ψ, T, srcs...)
        jointc = MI.classical_block_mutual_information(ψ, T, reduce(vcat, srcs))
        sync = MI.classical_synergy_information(ψ, T, srcs...)
        @printf("target %-2s: Q I(T:others)=%.4f Q synergy=%.4f | C I(T:others)=%.4f C synergy=%.4f\n",
            string(target), jointq, synq, jointc, sync)
    end
    @printf("Q CMI I(kx:ky|w)=%.4f  I(w:kx|ky)=%.4f  I(w:ky|kx)=%.4f\n",
        MI.conditional_mutual_information(ψ, KX, KY, W),
        MI.conditional_mutual_information(ψ, W, KX, KY),
        MI.conditional_mutual_information(ψ, W, KY, KX))
    @printf("C CMI I(kx:ky|w)=%.4f  I(w:kx|ky)=%.4f  I(w:ky|kx)=%.4f\n",
        MI.classical_conditional_mutual_information(ψ, KX, KY, W),
        MI.classical_conditional_mutual_information(ψ, W, KX, KY),
        MI.classical_conditional_mutual_information(ψ, W, KY, KX))
end

function print_diagonal_diag(Rd)
    groups, qf = diagonal_problem(Rd)
    R = 2Rd
    ψ = MI.amplitude_tensor(qf, fill(2, R)); ψ ./= norm(ψ)
    X,Y = groups[:x], groups[:y]
    println("\n=== Diagonal generalized MI Rd=$Rd R=$R ===")
    @printf("Q I(x:y)=%.4f   C I(x:y)=%.4f\n", MI.block_mutual_information(ψ, X, Y), MI.classical_block_mutual_information(ψ, X, Y))
    println("Top scale-resolved classical MI x_prefix : y_prefix:")
    for r in 1:Rd
        @printf("  r=%d C I(x1:%d : y1:%d)=%.4f Q=%.4f\n", r, r, r,
            MI.classical_block_mutual_information(ψ, X[1:r], Y[1:r]),
            MI.block_mutual_information(ψ, X[1:r], Y[1:r]))
    end
end

for Rd in 3:6
    print_green_diag(Rd)
end
print_diagonal_diag(6)
