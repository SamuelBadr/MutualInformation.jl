module MutualInformation

using LinearAlgebra

# Export main API function
export mutualinformation

# Export individual methods for advanced users
export mutualinformation_exact, mutualinformation_sampled

# Include implementation files
include("utils.jl")      # Shared utility functions
include("exact.jl")      # Exact computation methods
include("sampling.jl")   # Sampling-based approximation
include("api.jl")        # Unified API dispatcher

end # module MutualInformation
