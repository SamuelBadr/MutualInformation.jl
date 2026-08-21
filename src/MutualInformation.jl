module MutualInformation

using LinearAlgebra: eigvals!, Hermitian, tr
using Random: default_rng

# Export main API function
export mutualinformation

# Export individual methods for advanced users
export mutualinformation_exact, mutualinformation_uniform, mutualinformation_hybrid

# Backward compatibility (deprecated)
export mutualinformation_sampled  # Alias for mutualinformation_uniform

# Include implementation files
include("utils.jl")      # Shared utility functions
include("exact.jl")      # Exact computation methods
include("sampling.jl")   # Sampling-based approximation
include("api.jl")        # Unified API dispatcher

# Include OptimalCommunicationTree submodule
include("OptimalCommunicationTree/OptimalCommunicationTree.jl")
using .OptimalCommunicationTree
export OptimalCommunicationTree

include("quantics_layout.jl")  # Quantics bit-layout optimization from MI matrix
include("generalized_mi.jl")   # Block / conditional / higher-order MI diagnostics
include("layout_optimizer.jl") # Structured layout search utilities

# Export quantics layout API
export amplitude_tensor, bond_dimensions, cut_entropies,
       mi_ordering, minla_cost, path_to_ordering

# Export generalized MI diagnostics
export subsystem_entropy, block_mutual_information, conditional_mutual_information,
       interaction_information, synergy_information, total_correlation,
       block_mi_matrix,
       classical_subsystem_entropy, classical_block_mutual_information,
       classical_conditional_mutual_information, classical_synergy_information,
       classical_block_mi_matrix

# Export structured layout optimizer API
export BitLayoutSpec, LayoutCandidate, validate_candidate,
       coordinate_block_layouts, scale_interleavings, pair_interleave_layouts,
       sandwich_layouts, generate_layout_candidates,
       cut_weights, cutwidth_objective, layout_objectives,
       ExactTensorLayoutEvaluator, evaluate_layout, search_layouts,
       sketch_cut_score, beam_search_layout_by_cut_sketch

end # module MutualInformation
