import QuanticsGrids as QG
import MutualInformation as MI

# Define the quantics representation parameters
R = 13  # Number of qubits
grid = QG.DiscretizedGrid(R, -1.0, +1.0)

# Define the original function
f(x) = sin(200 * π * x) * exp(-x^2)  # Multi-scale: fast oscillations with Gaussian envelope

# Define the quantics version of the function
qf(qx) = f(QG.quantics_to_origcoord(grid, qx))

# Compute the mutual information matrix
println("Computing mutual information matrix for R=$R qubits...")
W = MI.mutualinformation(qf, R; method=:sampled, n_samples=1_000_000)

##

# Use the OptimalCommunicationTree submodule
using MutualInformation.OptimalCommunicationTree

result = solve_oct_problem(W, 2; max_iter=500000, initial_temp=100.0, final_temp=0.0001, verbose=true)#, use_random_init=true)
println("Best cost: ", result.cost)
println("Edges: ", result.edges)

# Load CairoMakie to enable plotting extension
using CairoMakie

# Plot the tree if plotting is available
plot_tree(result.tree, W; save_path="oct_tree.png", title="Optimal Communication Tree (max degree=100)")
