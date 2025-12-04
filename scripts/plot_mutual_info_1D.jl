#!/usr/bin/env julia

using Pkg
Pkg.activate(".")

using CairoMakie
import QuanticsGrids as QG
import MutualInformation as MI

# Define the quantics representation parameters
R = 9  # Number of qubits
grid = QG.DiscretizedGrid(R, -1.0, +1.0)

# Define the original function
f(x) = sin(200 * π * x) * exp(-x^2)  # Multi-scale: fast oscillations with Gaussian envelope

# Define the quantics version of the function
qf(qx) = f(QG.quantics_to_origcoord(grid, qx))

# Compute the mutual information matrix
# Uses unified API - automatically selects exact method since R=9 < 14
println("Computing mutual information matrix for R=$R qubits...")
# @time MI_matrix = MI.mutualinformation(qf, R; method=:sampled, n_samples=10_000)
MI_matrix = MI.mutualinformation(qf, R)#; method=:sampled, n_samples=10_000)

println("\nMutual Information Matrix:")
display(MI_matrix)
println()

# Create a heatmap visualization
fig = Figure(size=(800, 700))

ax = Axis(fig[1, 1],
    xlabel="Site B",
    ylabel="Site A",
    title="Mutual Information I(A:B) for f(x) = sin(20πx)·exp(-x²) [Log Scale]",
    aspect=DataAspect()
)

# Create a log-scale version of the matrix for plotting
# Replace zeros with a small value for log scale visualization
MI_matrix_log = copy(MI_matrix)
min_nonzero = minimum(MI_matrix[MI_matrix.>0])
epsilon = min_nonzero / 10  # Use value smaller than minimum
MI_matrix_log[MI_matrix_log.==0] .= epsilon

# Plot the heatmap with logarithmic color scale
hm = heatmap!(ax, MI_matrix_log,
    colormap=:viridis,
    colorscale=log10,
    colorrange=(epsilon, maximum(MI_matrix))
)

# Add colorbar with log scale
Colorbar(fig[1, 2], hm,
    label="Mutual Information (nats)",
    scale=log10,
    ticks=LogTicks(WilkinsonTicks(5))
)

# Add text annotations with values
# for i in 1:R
#     for j in 1:R
#         value = MI_matrix[i, j]
#         # Choose text color based on log-scale position
#         log_value = value > 0 ? log10(value) : log10(epsilon)
#         log_max = log10(maximum(MI_matrix))
#         log_min = log10(epsilon)
#         normalized_position = (log_value - log_min) / (log_max - log_min)
#         text_color = normalized_position > 0.5 ? :white : :black

#         text!(ax, j, i,
#             text=string(round(value, sigdigits=3)),
#             align=(:center, :center),
#             color=text_color,
#             fontsize=10
#         )
#     end
# end

# Set axis properties
ax.xticks = 1:R
ax.yticks = 1:R

# Save the figure
output_file = "mutual_information_matrix.png"
save(output_file, fig)
println("Plot saved to: $output_file")

# Also display the figure (if running in an interactive environment)
display(fig)

# Print some statistics
println("\nStatistics:")
println("  Maximum MI: $(maximum(MI_matrix)) nats")
println("  Minimum MI (off-diagonal): $(minimum(MI_matrix[MI_matrix .> 0]))")
println("  Mean MI (off-diagonal): $(sum(MI_matrix) / (R^2 - R))")
