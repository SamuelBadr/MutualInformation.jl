# Implementation Plan

## Goal
Build a systematic, reproducible quantics bit-layout optimizer that searches structured layout families and can optimize directly against exact/TCI bond dimensions, using pairwise MI only as one surrogate rather than the sole objective.

## Tasks
1. **Add core layout optimizer data structures**: Define explicit layout metadata, candidate, score, and result types so search logic is precise and testable.
   - File: `src/layout_optimizer.jl`
   - Changes:
     - Add `BitLayoutSpec` with fields like `R::Int`, `groups::Dict{Symbol,Vector{Int}}`, `scales::Dict{Int,Int}`, and `labels::Vector{String}`.
     - Add `LayoutCandidate(name::String, perm::Vector{Int}, family::Symbol, metadata::Dict{Symbol,Any})`.
     - Add `LayoutEvaluation(candidate, objectives, bond_profile, status, diagnostics)`.
     - Add validation helpers: `validate_layout_spec`, `validate_perm`, `canonical_perm_key`.
   - Acceptance: Unit tests reject duplicate/missing bits, invalid groups, non-permutations, and accept current Green's-function `{w,kx,ky}` spec.

2. **Wire optimizer into package exports without adding TCI as a hard dependency**: Keep core candidate generation and cheap objectives in the main package; keep TCI evaluation optional in experiments or a weak extension.
   - File: `src/MutualInformation.jl`
   - Changes:
     - `include("layout_optimizer.jl")` after `quantics_layout.jl`.
     - Export `BitLayoutSpec`, `LayoutCandidate`, `generate_layout_candidates`, `score_layouts`, `search_layouts`, `layout_objectives`.
   - Acceptance: `using MutualInformation` exposes core optimizer APIs and package still loads without `TensorCrossInterpolation`.

3. **Implement structured layout family generator**: Turn manual `green_layout_search.jl` ideas into reusable, exhaustive structured generators.
   - File: `src/layout_optimizer.jl`
   - Changes:
     - `coordinate_block_layouts(spec; all_group_orders=true, reverse_blocks=true)`.
     - `scale_interleavings(spec; group_orders=true, scale_directions=true)`.
     - `pair_interleave_layouts(spec; pair_orders=true, third_positions=(:before,:after), scale_directions=true)`.
     - `sandwich_layouts(spec; middle_groups=nothing, reverse_blocks=true)` that includes the Green winner `[kx..., reverse(w...), reverse(ky...)]`.
     - `segmented_block_layouts(spec; split_scales=[...])` for coarse/fine block splits such as `[kx_coarse, w_coarse, ky_coarse, kx_fine, w_fine, ky_fine]`.
     - `layout_family=:all` dispatcher returning de-duplicated candidates with stable names.
   - Acceptance: Tests show the generator includes blocked, interleaved, pair-interleaved, and Green sandwich candidates; all generated candidates are valid permutations and duplicates are removed.

4. **Add cheap surrogate objectives beyond MinLA**: Provide rigorous fast screens before expensive TCI evaluation.
   - File: `src/layout_optimizer.jl`
   - Changes:
     - `minla_objective(W, perm)` delegates to existing canonical `minla_cost`.
     - `cutwidth_objective(W, perm)` returns `(maximum_crossing_weight, sum_crossing_weight)` where crossing weight at cut `k` is `Σ_{i≤k<j} W[perm[i],perm[j]]`.
     - `bandwidth_objective(W, perm; threshold)` / weighted max distance for high-MI edges.
     - `scale_disruption_objective(spec, perm)` penalizes scrambled scale order within coordinate groups.
     - `group_boundary_objective(W, spec, perm)` estimates cross-group coupling across group boundaries.
     - `composite_surrogate_score(candidate; weights)` for ranking only, not final validation.
   - Acceptance: Known small matrices have hand-checked MinLA and cutwidth values; reversal invariance is tested where applicable.

5. **Add exact bond-dimension evaluator for small tensors**: Use existing `bond_dimensions`/`cut_entropies` as a first-class objective backend for small `R` and tests.
   - File: `src/layout_optimizer.jl`
   - Changes:
     - Define `AbstractLayoutEvaluator` and `ExactTensorEvaluator(ψ; rtol=1e-12)`.
     - `evaluate_layout(evaluator, candidate)` returns `χmax`, `χsum`, `Smax`, `Ssum`, `χ_profile`, `S_profile`.
     - Add lexicographic objective helper: `bond_objective(eval; mode=:lex_chimax_chisum)`.
   - Acceptance: Exact evaluator agrees with direct `bond_dimensions(ψ, perm)` and `cut_entropies(ψ, perm)` on Bell/GHZ and long-range pair tests.

6. **Add optional TCI bond-dimension evaluator in experiments**: Avoid a hard package dependency while making direct optimization against TCI systematic.
   - File: `experiments/tci_layout_evaluator.jl`
   - Changes:
     - Add `TCILayoutEvaluator(qf, R; tolerance=1e-8, maxiter=100, pivots=:deterministic, trials=1)`.
     - Implement `evaluate_layout(evaluator, candidate)` by constructing `g(u)=qf(u[invperm(candidate.perm)])`, running `TensorCrossInterpolation.crossinterpolate2`, and recording `χ_profile`, `χmax`, `χsum`, final error, elapsed time.
     - Add memoization keyed by `Tuple(candidate.perm)` and tolerance so repeated searches do not rerun TCI.
     - Deterministic pivots: `ones`, `twos`, alternating, plus optional seeded random pivots. For final validation use multiple pivot sets and report min/median/worst profiles.
   - Acceptance: A smoke script evaluates blocked and sandwich layouts and reproduces `χmax(blocked)≈264`, `χmax(sandwich)≈98` for `Rd=6`, `δ=0.2` within TCI tolerance variability.

7. **Implement multi-stage search driver**: Create a reproducible search pipeline that combines broad structured generation, cheap screening, diversity, and expensive direct evaluation.
   - File: `src/layout_optimizer.jl`
   - Changes:
     - `search_layouts(candidates; surrogate_scores, evaluator=nothing, top_k=..., diversity=:kendall, objective=:lex_chimax_chisum)`.
     - Stage A: generate all structured candidates.
     - Stage B: compute cheap objectives for every candidate.
     - Stage C: select diverse shortlist: top by each surrogate plus candidates with high Kendall/Hamming distance from already selected layouts.
     - Stage D: if evaluator provided, evaluate shortlist directly and rank by bond objective.
     - Stage E: local refinement around best structured candidates using grammar-preserving moves.
   - Acceptance: With a fake evaluator whose optimum is a known candidate, `search_layouts` finds that candidate; without evaluator it returns deterministic surrogate-ranked results.

8. **Implement grammar-preserving local refinement moves**: Optimize directly against exact/TCI objectives without exploring arbitrary random permutations first.
   - File: `src/layout_optimizer.jl`
   - Changes:
     - Moves: reverse a coordinate block, swap adjacent blocks, move one whole group between two groups, flip scale direction of a group, split a block at scale `s`, move a coarse/fine segment, convert two blocks into pair interleaving, convert pair interleaving back to blocks.
     - `refine_layout(candidate, evaluator; moves, max_evals, accept=:strict_lexicographic)`.
     - Include direct-TCI budget controls: `max_evals`, `max_seconds`, `early_stop_no_improve`.
   - Acceptance: On a synthetic exact tensor where pair adjacency is optimal, refinement improves a deliberately bad blocked layout to an interleaved/pair-adjacent layout within evaluation budget.

9. **Add multi-resolution/coarse-to-fine search**: Make Green-function layout discovery rigorous instead of overfitting `Rd=6`.
   - File: `experiments/green_layout_optimizer.jl`
   - Changes:
     - Run structured search at `Rd=3,4` with exact tensor evaluator where feasible.
     - Lift winning layout families to `Rd=5,6,7` by preserving grammar (`sandwich`, block order, reversals, split pattern).
     - Validate lifted candidates using TCI.
     - Report family stability table: candidate family, `Rd`, `χmax`, `χsum`, final TCI error.
   - Acceptance: Green sandwich family is selected or tied at small/coarse levels and validates at `Rd=6` with `χmax < blocked χmax` by a large margin.

10. **Add direct Green-function optimizer script**: Replace ad hoc `green_layout_search.jl` with a reproducible CLI-style script.
   - File: `experiments/green_layout_optimizer.jl`
   - Changes:
     - Build `{w,kx,ky}` `BitLayoutSpec` from `Rd`.
     - Generate layout families and surrogates.
     - Evaluate shortlist using `TCILayoutEvaluator` at coarse tolerance, then refine winners at `1e-8`.
     - Print and save `CSV`/`JSON` tables: all surrogate scores, evaluated candidates, winner profiles.
     - Include explicit baselines: original blocked, Z-order interleaved, MinLA, cutwidth, sandwich.
   - Acceptance: Running `julia -t auto --project=experiments experiments/green_layout_optimizer.jl --rd=6 --tol=1e-8` reports sandwich or better layout with `χmax <= 110` and blocked baseline around `χmax≈264`.

11. **Add a pairwise-positive benchmark script**: Keep the diagonal `exp(-(x-y)^2/σ^2)` case as a control where pairwise MI should work.
   - File: `experiments/diagonal_layout_optimizer.jl`
   - Changes:
     - Use same optimizer pipeline with exact evaluator for `Rd=6`.
     - Confirm interleaved or equivalent layout is found.
   - Acceptance: Optimizer reports `χmax=12` for interleaved/MI-style layouts vs blocked `χmax=28`, matching prior result.

12. **Add package tests for generators and search semantics**: Test the core optimizer without TCI dependency.
   - File: `test/test_layout_optimizer.jl`
   - Changes:
     - Layout spec validation tests.
     - Candidate family inclusion tests for blocked/interleaved/sandwich.
     - Objective tests for MinLA/cutwidth/bandwidth.
     - Exact evaluator tests on known small states.
     - Search-driver tests using fake evaluator and exact tensor evaluator.
   - Acceptance: `Pkg.test()` includes these tests and remains deterministic.

13. **Update `test/runtests.jl`**: Include the new optimizer tests.
   - File: `test/runtests.jl`
   - Changes: Add a `Testing Layout Optimizer` section that includes `test_layout_optimizer.jl`.
   - Acceptance: Full test suite runs all existing and new tests.

14. **Document optimizer concepts and limitations**: Make it clear that MI is a surrogate, while direct bond evaluation is the final objective.
   - File: `experiments/EXPERIMENTS.md`
   - Changes:
     - Add a section “Systematic optimizer” describing structured generation, surrogate screening, direct TCI evaluation, and multi-resolution validation.
     - Record Green's-function sandwich discovery as an optimizer result, not a manual one.
     - Explain when pairwise MI works and when direct/structured search is required.
   - Acceptance: Docs include exact commands and expected headline numbers for Green and diagonal benchmarks.

15. **Keep old exploratory scripts but mark them superseded or route to new optimizer**: Avoid two divergent search implementations.
   - Files: `experiments/green_layout_search.jl`, `experiments/experiment_2d_green.jl`
   - Changes:
     - Either convert `green_layout_search.jl` to call the new optimizer internals, or add a top comment “superseded by `green_layout_optimizer.jl`” and keep only for reproduction.
     - Update `experiment_2d_green.jl` to import/use `sandwich_perm` from shared helper or document it as a baseline.
   - Acceptance: No duplicated candidate-generation logic remains except in explicitly historical/reproduction scripts.

## Files to Modify
- `src/MutualInformation.jl` - include/export core layout optimizer APIs.
- `src/layout_optimizer.jl` - new core candidate generation, surrogate objectives, exact evaluator, search driver, local refinement.
- `test/runtests.jl` - include new optimizer tests.
- `test/test_layout_optimizer.jl` - new unit tests for specs, candidates, objectives, exact evaluator, search semantics.
- `experiments/tci_layout_evaluator.jl` - optional TCI evaluator and memoized bond-dimension evaluation.
- `experiments/green_layout_optimizer.jl` - systematic Green-function optimizer and validation script.
- `experiments/diagonal_layout_optimizer.jl` - positive-control optimizer script.
- `experiments/EXPERIMENTS.md` - update results and methodology.
- `experiments/green_layout_search.jl` - mark as superseded or refactor to call shared optimizer code.
- `experiments/experiment_2d_green.jl` - use shared layout helper/baselines or document as reproduction-only.

## New Files
- `src/layout_optimizer.jl` - core optimizer independent of TCI.
- `test/test_layout_optimizer.jl` - deterministic package tests for optimizer core.
- `experiments/tci_layout_evaluator.jl` - optional direct TCI bond-dimension evaluator.
- `experiments/green_layout_optimizer.jl` - end-to-end Green-function direct optimizer.
- `experiments/diagonal_layout_optimizer.jl` - end-to-end pairwise-correlation positive control.

## Dependencies
- Tasks 1-2 must come first because all other code depends on stable layout data structures and exports.
- Tasks 3-4 depend on Task 1 (`BitLayoutSpec`, `LayoutCandidate`).
- Task 5 depends on current `bond_dimensions`/`cut_entropies` and Task 1.
- Task 6 depends on experiments environment having `TensorCrossInterpolation`; it should not be added as a main package hard dependency.
- Tasks 7-8 depend on Tasks 3-6.
- Tasks 9-11 depend on Task 7 and Task 6 for TCI validation.
- Tasks 12-13 depend on Tasks 1-8.
- Tasks 14-15 depend on the scripts and results from Tasks 9-11.

## Risks
- **TCI stochastic/algorithmic variability**: Bond dimensions can vary with pivots and tolerance. Mitigation: deterministic pivot sets, repeated final validation for winners, record final errors and profiles.
- **Overfitting to Green at `Rd=6`**: A layout could win at one resolution only. Mitigation: multi-resolution `Rd=3,4,5,6` family stability tests and grammar-based lifting.
- **Search cost explosion**: Direct TCI evaluation is expensive. Mitigation: surrogate screening, diversity-limited shortlist, memoization, coarse-to-fine tolerance, explicit `max_evals`/`max_seconds`.
- **Core package dependency creep**: TCI should not become a hard dependency of `MutualInformation.jl`. Mitigation: keep TCI evaluator in `experiments/` or a future weak extension only.
- **Pairwise MI remains misleading for many-body functions**: This is expected. The optimizer must treat MI as a cheap feature, not the final objective.
- **Exact evaluator memory limits**: `amplitude_tensor` is exponential. Use exact evaluator only for small `R`; scripts should guard `R` and fall back to TCI.
- **Ambiguous objective**: Users may care about `χmax`, `χsum`, entropy, TCI error, or runtime. Mitigation: make objective lexicographic/configurable and always report the full profile.
- **Candidate grammar may miss better layouts**: Add local refinement moves and make candidate generators extensible; record unevaluated family counts.

## Validation Commands
1. `julia --project=. -t auto -e 'using Pkg; Pkg.test()'`
   - Expected: all existing tests plus `test_layout_optimizer.jl` pass deterministically.
2. `julia -t auto --project=experiments experiments/diagonal_layout_optimizer.jl --rd=6`
   - Expected: finds interleaved/equivalent layout with `χmax≈12` vs blocked `χmax≈28`.
3. `julia -t auto --project=experiments experiments/green_layout_optimizer.jl --rd=6 --tol=1e-8`
   - Expected: finds sandwich/equivalent layout with `χmax <= 110` vs blocked `χmax≈264`.
4. `julia -t auto --project=experiments experiments/green_layout_optimizer.jl --rd-min=3 --rd-max=6 --coarse-to-fine`
   - Expected: reports stable winning family across resolutions and validates lifted `Rd=6` result.

## Acceptance Report
```acceptance-report
{
  "criteriaSatisfied": [
    {
      "id": "criterion-1",
      "status": "satisfied",
      "evidence": "Created a scoped implementation plan only; it proposes a layout optimizer without adding unrelated features such as new MI estimators or non-path OCT solvers."
    }
  ],
  "changedFiles": [
    "plan.md"
  ],
  "testsAddedOrUpdated": [],
  "commandsRun": [
    {
      "command": "read /Users/samuel/Projects/MutualInformation.jl/context.md",
      "result": "failed",
      "summary": "Requested context file was missing (ENOENT); plan uses repository state and inherited context."
    },
    {
      "command": "ls /Users/samuel/Projects/MutualInformation.jl",
      "result": "passed",
      "summary": "Confirmed repository structure and available source/experiment/test directories."
    },
    {
      "command": "read Project.toml, src/MutualInformation.jl, src/quantics_layout.jl, src/OptimalCommunicationTree/tree_construction.jl, experiments/green_layout_search.jl, experiments/experiment_2d_green.jl, tests",
      "result": "passed",
      "summary": "Inspected existing package APIs, MinLA solver, layout helpers, manual Green search, experiment environment, and tests before planning."
    }
  ],
  "validationOutput": [
    "Planning-only run; no code tests executed by this subagent.",
    "context.md missing: ENOENT."
  ],
  "residualRisks": [
    "Plan depends on TCI behavior in experiments; implementation must validate tolerance/pivot stability.",
    "Working tree already contains inherited modified/untracked files unrelated to this planning artifact."
  ],
  "noStagedFiles": true,
  "diffSummary": "Added plan.md with concrete implementation plan for a structured/direct quantics layout optimizer.",
  "reviewFindings": [
    "no blockers in plan; context.md was missing and should be restored if it contains additional requirements"
  ],
  "manualNotes": "This subagent was instructed to plan only, so no source or test files were modified."
}
```
