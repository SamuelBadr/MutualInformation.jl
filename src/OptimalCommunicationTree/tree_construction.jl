using Random
using Graphs

"""
    oct_cost(tree::SimpleGraph, W::Matrix{Float64})

Compute the Optimal Communication Tree objective: ∑ᵢ<ⱼ W[i,j] * dist_T(i,j)
Assumes W is symmetric (W[i,j] = W[j,i]).
"""
function oct_cost(tree::SimpleGraph, W::Matrix{Float64})
    n = nv(tree)
    total = 0.0

    # BFS from each node to compute distances
    for src in 1:n
        distances = gdistances(tree, src)  # Built-in BFS from Graphs.jl

        # Sum over pairs (src, j) where j > src (assuming symmetric W)
        for j in (src+1):n
            total += W[src, j] * distances[j]
        end
    end

    return total
end

"""
    initial_tree_prim(n::Int, W::Matrix{Float64})

Create initial tree using Prim's algorithm: greedily add highest-weight edges.
Returns a SimpleGraph.
"""
function initial_tree_prim(n::Int, W::Matrix{Float64})
    tree = SimpleGraph(n)

    # Add isolated node 1 to start (degree will be 0 until first edge added)
    # We track "in tree" by checking if degree > 0 OR if it's node 1
    in_tree(v) = (v == 1 || degree(tree, v) > 0)

    # Build tree by repeatedly adding the highest-weight edge connecting tree to non-tree
    for _ in 1:(n-1)
        best_edge = (0, 0)
        best_weight = -Inf

        for u in 1:n, v in 1:n
            if in_tree(u) && !in_tree(v) && W[u, v] > best_weight
                best_weight = W[u, v]
                best_edge = (u, v)
            end
        end

        add_edge!(tree, best_edge...)
    end

    return tree
end

"""
    initial_tree_random(n::Int, max_deg::Int)

Create a random spanning tree respecting degree constraints.
Uses a simple random construction that adds edges uniformly at random.
Returns a SimpleGraph.
"""
function initial_tree_random(n::Int, max_deg::Int)
    tree = SimpleGraph(n)
    nodes_in_tree = Set([1])  # Start with node 1
    nodes_available = Set(2:n)

    while !isempty(nodes_available)
        # Pick a random node already in tree that has capacity
        candidates_in_tree = [u for u in nodes_in_tree if degree(tree, u) < max_deg]
        isempty(candidates_in_tree) && error("Cannot construct random tree with max_degree=$max_deg")

        u = rand(candidates_in_tree)

        # Pick a random node not yet in tree
        v = rand(collect(nodes_available))

        # Add edge
        add_edge!(tree, u, v)
        push!(nodes_in_tree, v)
        delete!(nodes_available, v)
    end

    return tree
end

"""
    get_components_after_removal(tree::SimpleGraph, u::Int, v::Int)

Find the two components that result from removing edge (u, v) from the tree.
Returns the two components as vectors of node indices.
"""
function get_components_after_removal(tree::SimpleGraph, u::Int, v::Int)
    # Create a copy without the edge
    tree_copy = copy(tree)
    successfully_removed = rem_edge!(tree_copy, u, v)
    successfully_removed || error("There was an error removing the edge ($u, $v) from $tree.")

    # Find connected components
    components = connected_components(tree_copy)
    length(components) == 2 || error("The graph $tree is not a tree.")

    return components[1], components[2]
end

"""
    find_swap_candidates(tree::SimpleGraph, comp1::Vector{Int}, comp2::Vector{Int}, max_deg::Int)

Find all edges that could reconnect components while respecting degree constraint.
"""
function find_swap_candidates(tree::SimpleGraph, comp1::Vector{Int}, comp2::Vector{Int}, max_deg::Int)
    candidates = Tuple{Int,Int}[]

    for u in comp1
        if degree(tree, u) >= max_deg
            continue  # Can't add more edges to u
        end
        for v in comp2
            if degree(tree, v) >= max_deg
                continue
            end
            push!(candidates, (u, v))
        end
    end

    return candidates
end

"""
    get_components_after_two_removals(tree::SimpleGraph, e1::Tuple{Int,Int}, e2::Tuple{Int,Int})

Find the components that result from removing two edges from the tree.
Returns a vector of component vectors (typically 3 components, or 2 if edges were adjacent).
"""
function get_components_after_two_removals(tree::SimpleGraph, e1::Tuple{Int,Int}, e2::Tuple{Int,Int})
    tree_copy = copy(tree)
    rem_edge!(tree_copy, e1...)
    rem_edge!(tree_copy, e2...)
    return connected_components(tree_copy)
end

"""
    find_two_edge_reconnections(tree::SimpleGraph, components::Vector{Vector{Int}}, max_deg::Int)

Find all valid ways to reconnect 3 components with 2 edges while respecting degree constraints.
Returns a vector of tuples: (edge1, edge2) where each edge is (u, v).
"""
function find_two_edge_reconnections(tree::SimpleGraph, components::Vector{Vector{Int}}, max_deg::Int)
    if length(components) != 3
        return Tuple{Tuple{Int,Int},Tuple{Int,Int}}[]
    end

    reconnections = Tuple{Tuple{Int,Int},Tuple{Int,Int}}[]
    comp1, comp2, comp3 = components

    # Try connecting: comp1-comp2 and comp2-comp3 (chain)
    for u1 in comp1, v2a in comp2
        degree(tree, u1) >= max_deg && continue
        degree(tree, v2a) >= max_deg && continue

        for v2b in comp2, u3 in comp3
            v2a == v2b && continue
            degree(tree, v2b) >= max_deg - 1 && continue  # v2b will get +1 from first edge
            degree(tree, u3) >= max_deg && continue

            push!(reconnections, ((u1, v2a), (v2b, u3)))
        end
    end

    # Try connecting: comp1-comp2 and comp1-comp3 (star with comp1 as center)
    for u1a in comp1, v2 in comp2
        degree(tree, u1a) >= max_deg && continue
        degree(tree, v2) >= max_deg && continue

        for u1b in comp1, v3 in comp3
            u1a == u1b && continue
            degree(tree, u1b) >= max_deg - 1 && continue
            degree(tree, v3) >= max_deg && continue

            push!(reconnections, ((u1a, v2), (u1b, v3)))
        end
    end

    # Try connecting: comp1-comp3 and comp2-comp3 (star with comp3 as center)
    for u1 in comp1, v3a in comp3
        degree(tree, u1) >= max_deg && continue
        degree(tree, v3a) >= max_deg && continue

        for u2 in comp2, v3b in comp3
            v3a == v3b && continue
            degree(tree, u2) >= max_deg && continue
            degree(tree, v3b) >= max_deg - 1 && continue

            push!(reconnections, ((u1, v3a), (u2, v3b)))
        end
    end

    return reconnections
end

"""
    try_one_edge_swap!(tree::SimpleGraph, W::Matrix{Float64}, max_deg::Int,
                       current_cost::Float64, temperature::Float64)

Attempt a single 1-edge swap move. Returns (accepted, new_cost, edges_to_remove, edges_to_add).
"""
function try_one_edge_swap!(tree::SimpleGraph, W::Matrix{Float64}, max_deg::Int,
    current_cost::Float64, temperature::Float64)
    edge_list = collect(edges(tree))
    shuffle!(edge_list)

    for edge in edge_list
        u_rem, v_rem = src(edge), dst(edge)

        # Get components after removing this edge
        comp1, comp2 = get_components_after_removal(tree, u_rem, v_rem)

        # Find candidate edges to add
        candidates = find_swap_candidates(tree, comp1, comp2, max_deg)
        isempty(candidates) && continue

        # Pick a random candidate for exploration
        u_add, v_add = rand(candidates)
        (u_add == u_rem && v_add == v_rem) && continue
        (u_add == v_rem && v_add == u_rem) && continue

        # Make the swap
        rem_edge!(tree, u_rem, v_rem)
        add_edge!(tree, u_add, v_add)

        # Evaluate
        new_cost = oct_cost(tree, W)
        Δcost = new_cost - current_cost

        # if Δcost < 0
        #     println("Found better config (1-edge swap)")
        # end

        # Metropolis acceptance criterion
        if Δcost < 0 || rand() < exp(-Δcost / temperature)
            # Accept the move
            return (true, new_cost)
        else
            # Reject: undo the swap
            rem_edge!(tree, u_add, v_add)
            add_edge!(tree, u_rem, v_rem)
        end
    end

    return (false, current_cost)
end

"""
    try_two_edge_swap!(tree::SimpleGraph, W::Matrix{Float64}, max_deg::Int,
                       current_cost::Float64, temperature::Float64)

Attempt a single 2-edge swap move. Returns (accepted, new_cost).
"""
function try_two_edge_swap!(tree::SimpleGraph, W::Matrix{Float64}, max_deg::Int,
    current_cost::Float64, temperature::Float64)
    edge_list = collect(edges(tree))
    shuffle!(edge_list)

    # Try a random pair of edges
    for _ in 1:min(10, length(edge_list))  # Limit attempts to avoid too much computation
        e1 = edge_list[rand(1:end)]
        e2 = edge_list[rand(1:end)]
        e1 == e2 && continue

        e1_tuple = (src(e1), dst(e1))
        e2_tuple = (src(e2), dst(e2))

        # Get components after removing both edges
        components = get_components_after_two_removals(tree, e1_tuple, e2_tuple)
        length(components) == 3 || continue

        # Find all ways to reconnect
        reconnections = find_two_edge_reconnections(tree, components, max_deg)
        isempty(reconnections) && continue

        # Pick a random reconnection
        (e_add1, e_add2) = rand(reconnections)

        # Make the swap
        rem_edge!(tree, e1_tuple...)
        rem_edge!(tree, e2_tuple...)
        add_edge!(tree, e_add1...)
        add_edge!(tree, e_add2...)

        # Evaluate
        new_cost = oct_cost(tree, W)
        Δcost = new_cost - current_cost

        # if Δcost < 0
        #     println("Found better config (2-edge swap)")
        # end

        # Metropolis acceptance criterion
        if Δcost < 0 || rand() < exp(-Δcost / temperature)
            # Accept the move
            return (true, new_cost)
        else
            # Reject: undo the swap
            rem_edge!(tree, e_add1...)
            rem_edge!(tree, e_add2...)
            add_edge!(tree, e1_tuple...)
            add_edge!(tree, e2_tuple...)
        end
    end

    return (false, current_cost)
end

"""
    solve_oct(W::Matrix{Float64}; max_deg::Int=typemax(Int),
              max_iter::Int=10000, verbose::Bool=false,
              initial_temp::Float64=1.0, final_temp::Float64=0.01,
              two_edge_prob::Float64=0.3, use_random_init::Bool=false)

Approximately solve the Optimal Communication Spanning Tree problem using simulated annealing.

Arguments:
- W: n×n matrix of non-negative communication weights
- max_deg: maximum allowed degree for any node (default: unconstrained)
- max_iter: maximum number of local search iterations
- verbose: print progress information
- initial_temp: starting temperature for simulated annealing
- final_temp: ending temperature for simulated annealing
- two_edge_prob: probability of attempting 2-edge swap vs 1-edge swap (default: 0.3)
- use_random_init: if true, start from random tree instead of Prim's (default: false)

Returns: (tree, cost) where tree is a SimpleGraph and cost is the objective value
"""
function solve_oct(W::Matrix{Float64}; max_deg::Int=typemax(Int),
    max_iter::Int=10000, verbose::Bool=false,
    initial_temp::Float64=1.0, final_temp::Float64=0.01,
    two_edge_prob::Float64=0.3, use_random_init::Bool=false)
    n = size(W, 1)
    @assert size(W, 2) == n "W must be square"
    @assert all(!isnegative, W) "W must have non-negative entries"
    @assert max_deg >= 2 || n <= 2 "max_deg must be at least 2 for n > 2"

    # Handle trivial cases
    if n <= 1
        return SimpleGraph(n), 0.0
    end
    if n == 2
        g = SimpleGraph(2)
        add_edge!(g, 1, 2)
        return g, W[1, 2]
    end

    # Create initial solution
    if use_random_init
        # Random initialization
        best_tree = initial_tree_random(n, max_deg)
        best_cost = oct_cost(best_tree, W)
        if verbose
            println("Random initial cost: $best_cost")
        end
    else
        # Smart initialization: try Prim, fall back to degree-constrained greedy
        tree_prim = initial_tree_prim(n, W)

        if maximum(degree(tree_prim)) <= max_deg
            # Prim solution respects degree constraint
            best_tree = tree_prim
        else
            # Prim violates constraint, use degree-constrained greedy
            best_tree = construct_degree_constrained_tree(n, W, max_deg)
        end
        best_cost = oct_cost(best_tree, W)

        if verbose
            println("Initial cost: $best_cost")
        end
    end

    # Simulated annealing: edge swap neighborhood
    tree = copy(best_tree)
    current_cost = best_cost

    # Exponential cooling schedule using logrange
    temperatures = logrange(initial_temp, final_temp, length=max_iter)

    for (iter, temperature) in enumerate(temperatures)
        # Decide whether to attempt 1-edge or 2-edge swap
        use_two_edge = rand() < two_edge_prob && ne(tree) >= 2

        accepted, new_cost = if use_two_edge
            try_two_edge_swap!(tree, W, max_deg, current_cost, temperature)
        else
            try_one_edge_swap!(tree, W, max_deg, current_cost, temperature)
        end

        if accepted
            current_cost = new_cost

            if current_cost < best_cost
                best_cost = current_cost
                best_tree = copy(tree)
                if verbose
                    move_type = use_two_edge ? "2-edge" : "1-edge"
                    println("Iter $iter (T=$(round(temperature, digits=4)), $move_type): improved to $best_cost")
                end
            end
        end
    end

    return best_tree, best_cost
end

"""
    construct_degree_constrained_tree(n::Int, W::Matrix{Float64}, max_deg::Int)

Construct a spanning tree respecting degree constraints using Kruskal's algorithm.
Returns a SimpleGraph.
"""
function construct_degree_constrained_tree(n::Int, W::Matrix{Float64}, max_deg::Int)
    tree = SimpleGraph(n)

    # Sort potential edges by communication weight (prefer high weight edges)
    all_edges = [(i, j, W[i, j]) for i in 1:n for j in (i+1):n]
    sort!(all_edges, by=x -> x[3], rev=true)

    for (u, v, _) in all_edges
        ne(tree) == n - 1 && break  # Have enough edges

        # Check degree constraints and connectivity
        if degree(tree, u) < max_deg && degree(tree, v) < max_deg
            # Adding this edge won't violate degree constraints
            # Check if it creates a cycle (would connect already-connected nodes)
            if !has_path(tree, u, v)
                add_edge!(tree, u, v)
            end
        end
    end

    # Validate that we constructed a valid spanning tree
    if ne(tree) != n - 1
        error("Cannot construct spanning tree with max_degree=$max_deg - constraint too restrictive. Only found $(ne(tree)) edges, need $(n-1).")
    end

    return tree
end

"""
    random_perturbation(tree::SimpleGraph, W::Matrix{Float64}, max_deg::Int)

Create a perturbed version of the tree for random restart.
"""
function random_perturbation(tree::SimpleGraph, W::Matrix{Float64}, max_deg::Int)
    new_tree = copy(tree)
    edge_list = collect(edges(new_tree))

    # Perform a few random valid swaps
    num_swaps = max(1, length(edge_list) ÷ 5)

    for _ in 1:num_swaps
        # Pick a random edge to remove
        edge = rand(edge_list)
        u_rem, v_rem = src(edge), dst(edge)

        # Get components and candidates after removal
        comp1, comp2 = get_components_after_removal(new_tree, u_rem, v_rem)
        candidates = find_swap_candidates(new_tree, comp1, comp2, max_deg)

        if !isempty(candidates)
            # Make a random swap
            u_add, v_add = rand(candidates)
            rem_edge!(new_tree, u_rem, v_rem)
            add_edge!(new_tree, u_add, v_add)

            # Update edge list
            filter!(e -> !(src(e) == u_rem && dst(e) == v_rem || src(e) == v_rem && dst(e) == u_rem), edge_list)
            push!(edge_list, Edge(u_add, v_add))
        end
    end

    cost = oct_cost(new_tree, W)
    return new_tree, cost
end

"""
    solve_oct_problem(W::Matrix{Float64}, max_degree_bound::Int; kwargs...)

Solve the Optimal Communication Spanning Tree problem.

Given communication demands W[i,j] between nodes, find a spanning tree T
minimizing ∑ᵢⱼ W[i,j] · dist_T(i,j) subject to maximum degree ≤ max_degree_bound.

Returns a named tuple (tree, cost, edges) containing the solution tree.
"""
function solve_oct_problem(W::Matrix{Float64}, max_degree_bound::Int; kwargs...)
    tree, cost = solve_oct(W; max_deg=max_degree_bound, kwargs...)
    edge_list = [(min(src(e), dst(e)), max(src(e), dst(e))) for e in edges(tree)]
    return (tree=tree, cost=cost, edges=edge_list)
end
