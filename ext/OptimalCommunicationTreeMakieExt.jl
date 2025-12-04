module OptimalCommunicationTreeMakieExt

using MutualInformation.OptimalCommunicationTree
using Graphs
using CairoMakie

"""
    plot_tree(tree::SimpleGraph, W::Matrix{Float64};
              node_labels=nothing, save_path="tree.png", title="Optimal Communication Tree")

Visualize the tree with edge weights mapped to line thickness.

# Arguments
- `tree`: A SimpleGraph representing the spanning tree
- `W`: Weight matrix (symmetric)
- `node_labels`: Optional vector of node labels (default: 1, 2, 3, ...)
- `save_path`: Where to save the figure (default: "tree.png")
- `title`: Plot title (default: "Optimal Communication Tree")

# Returns
- A Makie `Figure` object

# Example
```julia
using CairoMakie  # Must load CairoMakie to enable plotting
result = solve_oct_problem(W, 4; max_iter=5000, verbose=true)
fig = plot_tree(result.tree, W; save_path="my_tree.png")
```
"""
function OptimalCommunicationTree.plot_tree(tree::SimpleGraph, W::Matrix{Float64};
    node_labels=nothing, save_path="tree.png",
    title="Optimal Communication Tree")

    n = size(W, 1)

    # Default labels
    if node_labels === nothing
        node_labels = string.(1:n)
    end

    # Extract edges and weights
    edge_list = collect(edges(tree))
    edge_weights = [W[src(e), dst(e)] for e in edge_list]

    # Normalize edge weights for line thickness
    min_weight = minimum(edge_weights)
    max_weight = maximum(edge_weights)
    weight_range = max_weight - min_weight

    # Map weights to line widths (1.0 to 5.0)
    if weight_range > 1e-10
        line_widths = [1.0 + 4.0 * (w - min_weight) / weight_range for w in edge_weights]
    else
        line_widths = fill(3.0, length(edge_weights))
    end

    # Create figure with two columns: heatmap on left, tree on right
    fig = Figure(size=(1600, 800))

    # Left panel: Weight matrix heatmap with tree edges highlighted
    ax_heatmap = Axis(fig[1, 1],
        title="Weight Matrix with Tree Edges (log scale)",
        xlabel="Node", ylabel="Node",
        aspect=DataAspect())

    # Create log-scale matrix with black diagonal
    W_log = log10.(W .+ 1e-10)
    for i in 1:n
        W_log[i, i] = -Inf
    end

    # Draw heatmap of full matrix with log scale
    hm = heatmap!(ax_heatmap, 1:n, 1:n, W_log, colormap=:viridis)

    # Highlight tree edges with markers (both symmetric positions)
    edge_i = [coord for e in edge_list for coord in (src(e), dst(e))]
    edge_j = [coord for e in edge_list for coord in (dst(e), src(e))]

    # Plot tree edges as red squares
    scatter!(ax_heatmap, edge_i, edge_j,
        marker=:rect, markersize=20,
        color=:red, strokecolor=:white, strokewidth=2)

    # Add colorbar with log scale
    Colorbar(fig[1, 2], hm, label="log₁₀(Weight)")

    # Right panel: Tree visualization
    ax = Axis(fig[1, 3], title=title, aspect=DataAspect())
    hidedecorations!(ax)
    hidespines!(ax)

    # Compute node positions using tree layout
    positions = Vector{Point2f}(undef, n)

    # Simple tree layout algorithm (BFS-based)
    visited = falses(n)
    levels = zeros(Int, n)

    # Start from node with highest degree (likely root)
    root = argmax([degree(tree, i) for i in 1:n])

    queue = [root]
    visited[root] = true
    levels[root] = 0
    head = 1

    while head <= length(queue)
        u = queue[head]
        head += 1
        for v in neighbors(tree, u)
            if !visited[v]
                visited[v] = true
                levels[v] = levels[u] + 1
                push!(queue, v)
            end
        end
    end

    # Compute positions based on levels
    max_level = maximum(levels)
    level_positions = [Int[] for _ in 0:max_level]

    for i in 1:n
        push!(level_positions[levels[i]+1], i)
    end

    # Assign x, y coordinates
    for i in 1:n
        level = levels[i]
        nodes_at_level = level_positions[level+1]
        idx_in_level = findfirst(==(i), nodes_at_level)
        total_at_level = length(nodes_at_level)

        y = -Float32(level) * 2.0
        if total_at_level == 1
            x = 0.0f0
        else
            x = Float32((idx_in_level - 1) - (total_at_level - 1) / 2) * 2.0f0
        end

        positions[i] = Point2f(x, y)
    end

    # Draw edges with varying thickness
    for (edge_idx, e) in enumerate(edge_list)
        u, v = src(e), dst(e)
        pos_u = positions[u]
        pos_v = positions[v]

        lines!(ax, [pos_u[1], pos_v[1]], [pos_u[2], pos_v[2]],
            color=:gray40, linewidth=line_widths[edge_idx])

        # Add edge weight label
        mid_x = (pos_u[1] + pos_v[1]) / 2
        mid_y = (pos_u[2] + pos_v[2]) / 2

        text!(ax, mid_x, mid_y,
            text=string(round(edge_weights[edge_idx], digits=3)),
            align=(:center, :center),
            fontsize=9,
            color=:red)
    end

    # Draw nodes
    scatter!(ax, positions,
        color=:lightblue,
        strokecolor=:black,
        strokewidth=2,
        markersize=25)

    # Add node labels
    for i in 1:n
        text!(ax, positions[i][1], positions[i][2],
            text=node_labels[i],
            align=(:center, :center),
            fontsize=12,
            color=:black,
            font=:bold)
    end

    # Add legend spanning the full width
    Label(fig[2, 1:3],
        "Left: Weight matrix heatmap with tree edges marked in red | Right: Tree with edge thickness ∝ weight, red numbers = edge weights",
        tellwidth=false,
        fontsize=11)

    # Save figure
    save(save_path, fig)
    println("Tree visualization saved to: $save_path")

    return fig
end

end # module OptimalCommunicationTreeMakieExt
