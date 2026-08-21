using TensorCrossInterpolation: crossinterpolate2, linkdim

struct TCILayoutEvaluator{F}
    qf::F
    R::Int
    tolerance::Float64
    maxiter::Int
    pivots::Vector{Vector{Int}}
    cache::Dict{Tuple{Vararg{Int}},Any}
end

function TCILayoutEvaluator(qf, R::Int; tolerance=1e-8, maxiter=100,
    pivots=[ones(Int, R), fill(2, R)])
    return TCILayoutEvaluator(qf, R, Float64(tolerance), maxiter, pivots,
        Dict{Tuple{Vararg{Int}},Any}())
end

function MI.evaluate_layout(ev::TCILayoutEvaluator, cand::MI.LayoutCandidate)
    key = Tuple(cand.perm)
    if haskey(ev.cache, key)
        return ev.cache[key]
    end
    invp = invperm(cand.perm)
    g(u) = ev.qf(u[invp])
    tci, ranks, errors = crossinterpolate2(ComplexF64, g, fill(2, ev.R), ev.pivots;
        tolerance=ev.tolerance, maxiter=ev.maxiter)
    χ = [linkdim(tci, k) for k in 1:ev.R-1]
    result = (candidate=cand, χ=χ, S=Float64[], χmax=maximum(χ), χsum=sum(χ),
              Smax=NaN, Ssum=NaN, final_error=isempty(errors) ? NaN : errors[end])
    ev.cache[key] = result
    return result
end
