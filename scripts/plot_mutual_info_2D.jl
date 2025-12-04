#!/usr/bin/env julia

using Pkg
Pkg.activate(".")

using CairoMakie
import QuanticsGrids as QG
import MutualInformation as MI

# Define the quantics representation parameters
Rk = 10
Rnu = 10
w_inds = [[(:w, r)] for r in 1:Rnu]
x_inds = [[(:kx, r)] for r in 1:Rk]
y_inds = [[(:ky, r)] for r in 1:Rk]
indextable = [w_inds; x_inds; y_inds]
R = length(indextable)  # Number of qubits
# indextable = collect(Iterators.flatten(zip(x_inds, y_inds)))
wmax = 5.0
lower_bound = (-wmax, -1π, -1π)
upper_bound = (+wmax, +1π, +1π)
grid = QG.DiscretizedGrid((:w, :kx, :ky), indextable; lower_bound, upper_bound)

# Define the original function
disp(k) = -2 * sum(cos, k)
f(k; δ=0.1) = 1 / (k[1] - disp(k[2:end]) + im * δ)

# Define the quantics version of the function
qf(qx) = f(QG.quantics_to_origcoord(grid, qx))

# Compute the mutual information matrix
println("Computing mutual information matrix for R=$R qubits...")
MI_matrix = MI.mutualinformation(qf, R; method=:sampled, n_samples=10_000_000)

println("\nMutual Information Matrix:")
display(MI_matrix)
println()

##

# Create a heatmap visualization
fig = Figure(size=(800, 700))

ax = Axis(fig[1, 1],
    xlabel="Site B",
    ylabel="Site A",
    title="Mutual Information I(A:B) for G(w, kx, ky) [Log Scale]",
    aspect=DataAspect()
)

# Create a log-scale version of the matrix for plotting
# Replace zeros with a small value for log scale visualization
MI_matrix_log = copy(MI_matrix)
min_nonzero = minimum(MI_matrix[MI_matrix.>0])
epsilon = min_nonzero / 10  # Use value smaller than minimum
MI_matrix_log[MI_matrix_log.==0] .= epsilon

# for i in axes(MI_matrix_log, 1)
#     MI_matrix_log[i, i] = NaN
# end

# Plot the heatmap with logarithmic color scale
hm = heatmap!(ax, MI_matrix_log,
    colormap=:linear_tritanopic_krjcw_5_98_c46_n256,
    colorscale=log10,
    colorrange=(epsilon, maximum(MI_matrix))
)

# Add colorbar with log scale
Colorbar(fig[1, 2], hm,
    label="Mutual Information (nats)",
    # scale=log10,
    ticks=LogTicks(WilkinsonTicks(10))
)

# Set axis properties with labels from indextable
# Create labels from indextable structure
subscript_digits = ['₀', '₁', '₂', '₃', '₄', '₅', '₆', '₇', '₈', '₉']

# Helper function to convert number to subscript
to_subscript(n::Int) = join(subscript_digits[d+1] for d in reverse(digits(n)))

tick_labels = String[]
for idx_group in indextable
    # Each group is like [(:x, 1)] or [(:y, 2)]
    coord, level = idx_group[1]
    push!(tick_labels, string(coord) * to_subscript(level))
end

ax.xticks = (1:R, tick_labels)
ax.yticks = (1:R, tick_labels)

# Save the figure
output_file = "mutual_information_matrix.pdf"
save(output_file, fig)
println("Plot saved to: $output_file")

# Also display the figure (if running in an interactive environment)
display(fig)

# Print some statistics
println("\nStatistics:")
println("  Maximum MI: $(maximum(MI_matrix)) nats")
println("  Minimum MI (off-diagonal): $(minimum(MI_matrix[MI_matrix .> 0]))")
println("  Mean MI (off-diagonal): $(sum(MI_matrix) / (R^2 - R))")
