using Pkg; Pkg.activate(@__DIR__)
using DelimitedFiles, Printf, LinearAlgebra
W = readdlm(joinpath(@__DIR__, "out_2d_MI_matrix.csv"), ',', Float64)
R = size(W, 1)
Rd = 6
labels = vcat(fill("w ", Rd), fill("kx", Rd), fill("ky", Rd))
println("R=$R  total MI (off-diag) = ", round(sum(W) / 2, digits=3))

function blockmi(a, b)
    s = 0.0; n = 0
    for i in a, j in b
        if i != j; s += W[i, j]; n += 1; end
    end
    return n > 0 ? s / n : 0.0
end
w = 1:6; kx = 7:12; ky = 13:18
println("\nmean MI within w-w : ", round(blockmi(w, w), digits=4))
println("mean MI within kx-kx: ", round(blockmi(kx, kx), digits=4))
println("mean MI within ky-ky: ", round(blockmi(ky, ky), digits=4))
println("mean MI w-kx        : ", round(blockmi(w, kx), digits=4))
println("mean MI w-ky        : ", round(blockmi(w, ky), digits=4))
println("mean MI kx-ky       : ", round(blockmi(kx, ky), digits=4))

println("\nmax off-diag MI = ", round(maximum(W[i, j] for i in 1:R for j in 1:R if i != j), digits=4))
pairs = sort([(W[i, j], i, j) for i in 1:R for j in (i + 1):R], rev=true)[1:10]
println("top MI pairs (val, i, j):")
for (v, i, j) in pairs
    println("  ", round(v, digits=4), "  bit ", i, "(", labels[i], ") <-> bit ", j, "(", labels[j], ")")
end
