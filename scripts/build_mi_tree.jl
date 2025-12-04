#!/usr/bin/env julia
#
# build_mi_tree.jl
#
# Constructs an optimal tree topology from a mutual information matrix
# using a TN-optimized greedy merging approach. The goal is to place
# highly correlated qubits close together in the tree structure.
#
# This is useful for tensor network contraction ordering, where having
# strongly correlated subsystems adjacent in the tree minimizes
# entanglement across cuts.
#
# Usage: julia --project=. scripts/build_mi_tree.jl

using Graphs
import MutualInformation as MI

# Try to load plotting libraries (optional)
const PLOTTING_AVAILABLE = try
    @eval using CairoMakie
    @eval using NetworkLayout
    true
catch e
    @warn "Plotting libraries not available: $e\nTree visualization will be skipped."
    false
end


"""
    compute_weighted_path_cost(tree::SimpleGraph, MI_matrix::Matrix{Float64}) -> Float64

Computes the weighted path distance cost for a tree.

Cost = Σ(i<j) MI(i,j) × distance_tree(i,j)

where distance_tree(i,j) is the number of hops between nodes i and j in the tree.
This measures how well the tree structure matches the MI correlations - lower is better.

# Arguments
- `tree`: A tree graph
- `MI_matrix`: n×n symmetric matrix of mutual information values

# Returns
- The total weighted path distance cost
"""
function compute_weighted_path_cost(tree::SimpleGraph, MI_matrix::Matrix{Float64})
    n = nv(tree)
    total_cost = 0.0

    # Compute all-pairs shortest paths using BFS from each node
    for i in 1:n
        # BFS to find distances from node i to all others
        distances = fill(-1, n)
        distances[i] = 0
        queue = [i]
        head = 1

        while head <= length(queue)
            current = queue[head]
            head += 1

            for neighbor in neighbors(tree, current)
                if distances[neighbor] == -1
                    distances[neighbor] = distances[current] + 1
                    push!(queue, neighbor)
                end
            end
        end

        # Add weighted distances for pairs (i, j) where j > i
        for j in (i+1):n
            total_cost += MI_matrix[i, j] * distances[j]
        end
    end

    return total_cost
end


"""
    build_tn_optimized_tree(MI_matrix::Matrix{Float64}; max_degree::Int=typemax(Int)) -> SimpleGraph{Int}

Constructs a tree topology from a mutual information matrix by minimizing
the weighted path distance cost function, subject to a maximum degree constraint.

The algorithm:
1. Start with each qubit as its own tree (singleton nodes)
2. Repeatedly merge the two trees that minimize the weighted path distance cost
3. Continue until all nodes are in a single tree

# Arguments
- `MI_matrix`: n×n symmetric matrix of mutual information values
- `max_degree`: Maximum allowed degree for any node (default: no limit)

# Returns
- A `SimpleGraph` representing the tree structure where each node is a qubit

# Algorithm Details
Cost function: Σ(i<j) MI(i,j) × distance_tree(i,j)

At each step, we:
- Consider all possible merges between pairs of trees
- For each merge, consider all possible connecting edges that don't violate max_degree
- Evaluate the resulting weighted path distance cost
- Choose the merge that results in the lowest total cost

The max_degree constraint prevents "star" topologies with high-degree hub nodes,
which are computationally expensive for tensor network contraction.

# Example
```julia
MI_matrix = [0.0  0.8  0.2  0.1;
             0.8  0.0  0.7  0.1;
             0.2  0.7  0.0  0.9;
             0.1  0.1  0.9  0.0]
tree = build_tn_optimized_tree(MI_matrix; max_degree=3)
```
"""
function build_tn_optimized_tree(MI_matrix::Matrix{Float64}; max_degree::Int=typemax(Int))
    n = size(MI_matrix, 1)

    # Validate max_degree
    if max_degree < 1
        error("max_degree must be at least 1")
    end
    if max_degree == 1 && n > 2
        error("max_degree=1 cannot form a tree with n>2 nodes")
    end

    # Initialize: each node is its own tree
    tree_id = collect(1:n)  # tree_id[i] = which tree node i belongs to
    trees = [Set([i]) for i in 1:n]  # trees[k] = set of nodes in tree k

    # Initialize the graph (initially no edges)
    tree_graph = SimpleGraph(n)

    # We need to merge n-1 times to get a single tree
    for _ in 1:(n-1)
        # Find the best merge that minimizes the weighted path distance cost
        best_cost = Inf
        best_edge = (0, 0)

        # Consider all pairs of distinct trees
        active_trees = unique(tree_id)

        for (idx_a, tree_a_id) in enumerate(active_trees)
            for tree_b_id in active_trees[(idx_a+1):end]
                # Get nodes in each tree
                nodes_a = trees[tree_a_id]
                nodes_b = trees[tree_b_id]

                # Try all possible edges connecting these two trees
                for node_a in nodes_a
                    for node_b in nodes_b
                        # Check degree constraint: would this edge violate max_degree?
                        current_degree_a = degree(tree_graph, node_a)
                        current_degree_b = degree(tree_graph, node_b)

                        if current_degree_a >= max_degree || current_degree_b >= max_degree
                            # Skip this edge - it would violate the degree constraint
                            continue
                        end

                        # Create a temporary graph with this edge added
                        temp_graph = copy(tree_graph)
                        add_edge!(temp_graph, node_a, node_b)

                        # Compute the cost of this configuration
                        cost = compute_weighted_path_cost(temp_graph, MI_matrix)

                        # Update best if this is better
                        if cost < best_cost
                            best_cost = cost
                            best_edge = (node_a, node_b)
                        end
                    end
                end
            end
        end

        # Check if we found a valid edge
        if best_edge == (0, 0)
            error("Cannot construct tree with max_degree=$max_degree - constraint too restrictive")
        end

        # Add the best edge to the graph
        node_a, node_b = best_edge
        add_edge!(tree_graph, node_a, node_b)

        # Update tree membership
        # Find which trees these nodes belong to
        tree_a_id = tree_id[node_a]
        tree_b_id = tree_id[node_b]

        # Merge tree_b into tree_a
        for node in trees[tree_b_id]
            tree_id[node] = tree_a_id
            push!(trees[tree_a_id], node)
        end
        empty!(trees[tree_b_id])
    end

    return tree_graph
end


"""
    print_tree_structure(tree::SimpleGraph, MI_matrix::Matrix{Float64};
                        node_labels=nothing)

Prints the tree structure in a readable text format.

# Arguments
- `tree`: Tree graph from `build_tn_optimized_tree`
- `MI_matrix`: Original mutual information matrix
- `node_labels`: Optional vector of node labels (default: 1, 2, 3, ...)
"""
function print_tree_structure(tree::SimpleGraph, MI_matrix::Matrix{Float64};
    node_labels=nothing)
    n = nv(tree)

    # Default labels
    if node_labels === nothing
        node_labels = string.(1:n)
    end

    println("\n" * "="^70)
    println("Tree Edges (node1 -- node2 : MI value)")
    println("="^70)

    # Sort edges by MI value (descending)
    edge_list = collect(edges(tree))
    edge_mis = [MI_matrix[src(e), dst(e)] for e in edge_list]
    perm = sortperm(edge_mis, rev=true)

    for idx in perm
        edge = edge_list[idx]
        i, j = src(edge), dst(edge)
        mi_val = edge_mis[idx]
        println("  $(node_labels[i]) -- $(node_labels[j]) : $(round(mi_val, digits=4)) nats")
    end
    println("="^70)
end


"""
    plot_mi_tree(tree::SimpleGraph, MI_matrix::Matrix{Float64};
                 node_labels=nothing,
                 save_path="mi_tree.png")

Visualizes the MI-based tree structure with edge weights shown.
Only available if CairoMakie and NetworkLayout are loaded.

# Arguments
- `tree`: Tree graph from `build_tn_optimized_tree`
- `MI_matrix`: Original mutual information matrix
- `node_labels`: Optional vector of node labels (default: 1, 2, 3, ...)
- `save_path`: Where to save the figure (default: "mi_tree.png")

# Returns
- A Makie `Figure` object, or `nothing` if plotting is unavailable

# Example
```julia
tree = build_tn_optimized_tree(MI_matrix)
fig = plot_mi_tree(tree, MI_matrix; save_path="my_tree.png")
```
"""
function plot_mi_tree(tree::SimpleGraph, MI_matrix::Matrix{Float64};
    node_labels=nothing,
    save_path="mi_tree.png")
    if !PLOTTING_AVAILABLE
        @warn "Plotting not available. Skipping visualization."
        return nothing
    end

    n = nv(tree)

    # Default labels
    if node_labels === nothing
        node_labels = string.(1:n)
    end

    # Compute layout using Spring/Stress layout for trees
    layout = Spring()(tree)

    # Extract node positions
    node_x = [p[1] for p in layout]
    node_y = [p[2] for p in layout]

    # Create figure
    fig = Figure(size=(1000, 800))
    ax = Axis(fig[1, 1],
        title="TN-Optimized Tree from Mutual Information",
        aspect=DataAspect()
    )
    hidedecorations!(ax)
    hidespines!(ax)

    # Draw edges with thickness proportional to MI
    edge_mis = Float64[]
    for edge in edges(tree)
        i, j = src(edge), dst(edge)
        mi_val = MI_matrix[i, j]
        push!(edge_mis, mi_val)
    end

    # Normalize edge widths
    max_mi = maximum(edge_mis)
    min_mi = minimum(edge_mis)

    for (edge, mi_val) in zip(edges(tree), edge_mis)
        i, j = src(edge), dst(edge)

        # Edge width proportional to MI
        width = 1.0 + 4.0 * (mi_val - min_mi) / (max_mi - min_mi + 1e-10)

        # Draw edge
        lines!(ax, [node_x[i], node_x[j]], [node_y[i], node_y[j]],
            color=:gray40, linewidth=width)

        # Add edge label (MI value)
        mid_x = (node_x[i] + node_x[j]) / 2
        mid_y = (node_y[i] + node_y[j]) / 2
        text!(ax, mid_x, mid_y,
            text=string(round(mi_val, digits=3)),
            align=(:center, :center),
            fontsize=10,
            color=:red)
    end

    # Draw nodes
    scatter!(ax, node_x, node_y,
        color=:lightblue,
        strokecolor=:black,
        strokewidth=2,
        markersize=30)

    # Add node labels
    for (i, label) in enumerate(node_labels)
        text!(ax, node_x[i], node_y[i],
            text=label,
            align=(:center, :center),
            fontsize=14,
            color=:black,
            font=:bold)
    end

    # Add legend explaining edge thickness
    Label(fig[2, 1],
        "Edge thickness ∝ Mutual Information\nRed numbers = MI values (nats)",
        tellwidth=false,
        fontsize=12)

    # Save figure
    save(save_path, fig)
    println("Tree visualization saved to: $save_path")

    return fig
end


"""
    analyze_tree_structure(tree::SimpleGraph, MI_matrix::Matrix{Float64})

Prints statistics about the constructed tree.

# Arguments
- `tree`: Tree graph from `build_tn_optimized_tree`
- `MI_matrix`: Original mutual information matrix
"""
function analyze_tree_structure(tree::SimpleGraph, MI_matrix::Matrix{Float64})
    n = nv(tree)

    println("\n" * "="^70)
    println("Tree Structure Analysis")
    println("="^70)

    # Basic properties
    println("Number of nodes: $n")
    println("Number of edges: $(ne(tree))")

    # Edge MI statistics
    edge_mis = [MI_matrix[src(e), dst(e)] for e in edges(tree)]
    println("\nEdge Mutual Information:")
    println("  Maximum: $(round(maximum(edge_mis), digits=4)) nats")
    println("  Minimum: $(round(minimum(edge_mis), digits=4)) nats")
    println("  Mean:    $(round(sum(edge_mis) / length(edge_mis), digits=4)) nats")
    println("  Total:   $(round(sum(edge_mis), digits=4)) nats")

    # Compare to all possible edges
    all_pairs_mi = [MI_matrix[i, j] for i in 1:n for j in i+1:n]
    println("\nAll pairwise MI (for comparison):")
    println("  Maximum: $(round(maximum(all_pairs_mi), digits=4)) nats")
    println("  Mean:    $(round(sum(all_pairs_mi) / length(all_pairs_mi), digits=4)) nats")

    # Efficiency metric: what fraction of total MI is captured in tree edges?
    tree_mi_total = sum(edge_mis)
    all_mi_total = sum(all_pairs_mi)
    efficiency = tree_mi_total / all_mi_total * 100
    println("\nTree efficiency: $(round(efficiency, digits=2))%")
    println("(Percentage of total MI captured by tree edges)")

    # Node degrees
    degrees = degree(tree)
    println("\nNode degrees:")
    println("  Maximum degree: $(maximum(degrees))")
    println("  Minimum degree: $(minimum(degrees))")
    println("  Mean degree:    $(round(sum(degrees) / n, digits=2))")

    # Identify leaves and internal nodes
    leaves = [i for i in 1:n if degrees[i] == 1]
    println("\nLeaf nodes (degree 1): $(length(leaves))")
    println("Internal nodes: $(n - length(leaves))")

    println("="^70 * "\n")
end


#=============================================================================
                                EXAMPLE USAGE
=============================================================================#

function test_tree_structure()
    println("="^70)
    println("TN-Optimized Tree Construction from MI Matrix")
    println("="^70)

    # Example 1: Small predefined MI matrix
    println("\n--- Example 1: 5-qubit system ---\n")

    # Create a test MI matrix with clear structure:
    # Qubits 1-2-3 form a chain, 4-5 form a pair
    MI_test = [
        0.0 0.9 0.3 0.1 0.1;
        0.9 0.0 0.8 0.1 0.1;
        0.3 0.8 0.0 0.2 0.2;
        0.1 0.1 0.2 0.0 0.7;
        0.1 0.1 0.2 0.7 0.0
    ]

    println("Input MI Matrix:")
    display(MI_test)

    # Build tree
    tree = build_tn_optimized_tree(MI_test)

    # Analyze and print
    analyze_tree_structure(tree, MI_test)
    print_tree_structure(tree, MI_test; node_labels=["Q$i" for i in 1:5])

    # Visualize if available
    if PLOTTING_AVAILABLE
        fig1 = plot_mi_tree(tree, MI_test;
            node_labels=["Q$i" for i in 1:5],
            save_path="mi_tree_example1.png")
        display(fig1)
    end


    # Example 2: Compute from a real wavefunction
    println("\n--- Example 2: 8-qubit nearest-neighbor correlated system ---\n")

    # Define a wavefunction with nearest-neighbor correlations
    function f_nn_correlated(x)
        # Exponential preference for matching neighbors
        correlation = sum(x[i] == x[i+1] ? 1.0 : 0.0 for i in 1:length(x)-1)
        return exp(correlation / 2)
    end

    # Compute MI matrix (using exact method for len=8)
    len = 8
    println("Computing MI matrix for $len qubits...")
    @time MI_matrix = MI.mutualinformation(f_nn_correlated, len)

    println("\nMI Matrix:")
    display(round.(MI_matrix, digits=3))

    # Build tree
    println("\nBuilding TN-optimized tree...")
    tree = build_tn_optimized_tree(MI_matrix)

    # Analyze and print
    analyze_tree_structure(tree, MI_matrix)
    print_tree_structure(tree, MI_matrix; node_labels=["Q$i" for i in 1:len])

    # Visualize if available
    if PLOTTING_AVAILABLE
        fig2 = plot_mi_tree(tree, MI_matrix;
            node_labels=["Q$i" for i in 1:len],
            save_path="mi_tree_example2.png")
        display(fig2)
    end


    # Example 3: Same system but with max_degree constraint
    println("\n--- Example 3: Same system with max_degree=3 constraint ---\n")

    println("Building TN-optimized tree with max_degree=3...")
    tree_constrained = build_tn_optimized_tree(MI_matrix; max_degree=3)

    # Analyze and print
    analyze_tree_structure(tree_constrained, MI_matrix)
    print_tree_structure(tree_constrained, MI_matrix; node_labels=["Q$i" for i in 1:len])

    # Visualize if available
    if PLOTTING_AVAILABLE
        fig3 = plot_mi_tree(tree_constrained, MI_matrix;
            node_labels=["Q$i" for i in 1:len],
            save_path="mi_tree_example3_maxdeg3.png")
        display(fig3)
    end

    # Compare costs
    cost_unconstrained = compute_weighted_path_cost(tree, MI_matrix)
    cost_constrained = compute_weighted_path_cost(tree_constrained, MI_matrix)
    println("\nCost comparison:")
    println("  Unconstrained tree: $(round(cost_unconstrained, digits=4))")
    println("  Max-degree-3 tree:  $(round(cost_constrained, digits=4))")
    println("  Cost increase:      $(round((cost_constrained - cost_unconstrained) / cost_unconstrained * 100, digits=2))%")


    println("\n" * "="^70)
    if PLOTTING_AVAILABLE
        println("Done! Check the generated PNG files for visualizations.")
    else
        println("Done! (Visualization skipped - plotting libraries unavailable)")
    end
    println("="^70)
end


# Run the examples if this script is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    test_tree_structure()
end
