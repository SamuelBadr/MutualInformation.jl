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

end # module MutualInformation
