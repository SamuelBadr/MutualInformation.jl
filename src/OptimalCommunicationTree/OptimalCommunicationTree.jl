"""
    OptimalCommunicationTree

Submodule for solving the Optimal Communication Spanning Tree problem.

Given a symmetric weight matrix W representing communication demands between nodes,
finds a spanning tree T that minimizes ∑ᵢⱼ W[i,j] · dist_T(i,j), where dist_T(i,j)
is the distance between nodes i and j in the tree.

The trusted supported case is `max_deg=2`, which is exactly the weighted
Minimum Linear Arrangement / Hamiltonian path problem used for MPS bit layouts.
For this case the module uses exact enumeration on small instances and a
multi-start permutation local search on larger instances.

## Visualization

The `plot_tree` function is available as a package extension and requires loading CairoMakie:
```julia
using CairoMakie
plot_tree(tree, W; save_path="tree.png")
```
"""
module OptimalCommunicationTree

using Graphs

# Export main functions
export solve_oct_problem, solve_oct, solve_minla, exact_minla, heuristic_minla
export oct_cost, minla_cost, path_graph_from_ordering, ordering_from_path
export plot_tree

# Include implementation files
include("tree_construction.jl")

# Declare plot_tree function (implemented in extension)
"""
    plot_tree(tree::SimpleGraph, W::Matrix{Float64}; kwargs...)

Visualize the optimal communication tree with edge weights.

**Note:** This function requires CairoMakie to be loaded.
Load it with `using CairoMakie` before calling this function.

# Arguments
- `tree`: A SimpleGraph representing the spanning tree
- `W`: Weight matrix (symmetric)
- `node_labels`: Optional vector of node labels (default: 1, 2, 3, ...)
- `save_path`: Where to save the figure (default: "tree.png")
- `title`: Plot title (default: "Optimal Communication Tree")

# Example
```julia
using CairoMakie  # Required!
result = solve_oct_problem(W, 2; max_iter=5000, verbose=true)
fig = plot_tree(result.tree, W; save_path="my_path.png")
```
"""
function plot_tree end

end # module OptimalCommunicationTree
