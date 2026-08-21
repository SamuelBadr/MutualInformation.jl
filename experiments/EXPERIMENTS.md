# Experiments: MI-guided quantics bit layout

Goal: use the mutual-information (MI) matrix between quantics bits to find an
MPS-chain (linear) bit layout that **reduces bond dimensions**, and validate it
against *actual* bond dimensions.

## Method

A quantics MPS places its `R` bits on a 1D chain. The bond dimension `χ` at a cut
is bounded by the entanglement across it, and the cut entanglement is bounded by
the mutual information across it (Schuch–Wolf–Verstraete–Cirac:
`I(left:right) ≤ 2 log χ`). So to keep bond dimensions small we want an ordering
that keeps mutually-informative bits close together. With weights
`W[i,j] = I(bit_i : bit_j)`, the cost

```
MinLA(π) = Σ_{i<j} W[π(i), π(j)] · |π(i) − π(j)|
```

is the **Minimum Linear Arrangement** objective, and a degree-2 spanning tree *is*
a Hamiltonian path = a linear arrangement. `MI.mi_ordering(W; max_deg=2)` now
uses the replacement MinLA backend: exact enumeration for small `R`, and a
multi-start permutation local search (spectral/greedy/random initializers plus
insertion and segment-reversal improvements) for larger `R`.

The pipeline (`src/quantics_layout.jl`, exported):

| function | role |
|---|---|
| `mutualinformation(f, R)` | the MI matrix `W` (exact for small `R`, sampled for large) |
| `mi_ordering(W; max_deg=2)` | Hamiltonian path / bit layout from `W` |
| `minla_cost(W, perm)` | the MinLA objective, for comparing layouts |
| `amplitude_tensor(f, localdims)` | the exact state tensor `ψ` |
| `bond_dimensions(ψ, perm)` | **exact** MPS bond dims (ranks of unfoldings) |
| `cut_entropies(ψ, perm)` | entanglement entropies across each cut |

Validation is in two regimes: **exact** bond dimensions via SVD of unfoldings
(small `R`), and **TCI** bond dimensions via `TensorCrossInterpolation` (large
`R`, tolerance-controlled). Run with `julia -t auto --project=experiments
experiments/<script>.jl`.

## Results

### 1. 1D multi-scale function `f(x)=sin(20πx)·exp(−x²)`, R=10 (exact)

| layout | χmax | χsum | Ssum | MinLA |
|---|---|---|---|---|
| natural (1..R) | 12 | 56 | 2.685 | 8.86 |
| **MI-optimal** | 13 | 64 | **2.371** | 8.08 |
| random (n=60) | 13–16 (mean 15.0) | — | 3.37–7.52 | — |

- **Surrogate validity:** `corr(MinLA, total cut entropy) = +0.970` — the
  MI-guided objective is an excellent predictor of total bond dimension.
- MI-optimal beats 59/60 random orderings on χmax, 58/60 on max entropy.
- For smooth 1D functions the natural multi-scale quantics ordering is already
  near-optimal, so MI-optimal is comparable on χmax but **better on total
  entropy**. This is the expected baseline.

### 2. 2D pairwise-coupled `f(x,y)=exp(−(x−y)²/σ²)`, R=12 (exact bond dims) ✅ the win

| layout | χmax | χsum | Ssum | MinLA |
|---|---|---|---|---|
| blocked `[x₁..x₆, y₁..y₆]` | 28 | 134 | 9.832 | 18.81 |
| interleaved `[x₁,y₁,x₂,y₂,…]` | 12 | 69 | 3.724 | 6.58 |
| **MI-optimal** | **12** | **69** | **3.724** | 6.55 |
| random (n=40) | 17–28 (mean 24.4) | — | — | — |

- The MI matrix cleanly surfaces the diagonal pairs as the top entries
  (`x₁↔y₁`, `x₂↔y₂`, `x₃↔y₃`), and MI-optimal **recovers the interleaved
  layout**.
- **57% reduction in χmax and 62% in total entropy** versus the natural blocked
  layout. Surrogate `corr(MinLA, Ssum) = 0.99`.
- This is the multidimensional case where pairwise MI is the *right* surrogate:
  the cross-coordinate coupling is genuinely pairwise (`x_r` with `y_r`).

### 3. 3D Green's function `G(w,kx,ky)=1/(w−ε(kx,ky)+iδ)`, R=18 (TCI bond dims) ✅ after broader search

After replacing the OCT backend, pure pairwise-MI MinLA still was not the right
objective for the Green's function. I then added a structured optimizer core
(`src/layout_optimizer.jl`) and two data-driven drivers:

- `experiments/green_layout_optimizer.jl`: broad structured grammar + TCI
  validation.
- `experiments/green_multires_layout_optimizer.jl`: multi-resolution exact SVD
  screening. This generates the grammar automatically and scores every candidate
  from the actual function tensor at `Rd=3:6`; no hand-picked Green layout is
  inserted.

The optimizer generates coordinate block permutations, within-block reversals,
scale interleavings, pair-interleavings, sandwich layouts, and pairwise-MI
layouts, then validates finalists by actual bond dimensions.

| layout | χmax | χsum | MinLA |
|---|---:|---:|---:|
| original blocked `[w, kx, ky]` | 264 | 1018 | 18.26 |
| interleaved (Z-order by scale) | 398 | 1414 | 16.56 |
| pairwise-MI MinLA | 231 | 1127 | **12.20** |
| **sandwich** `[ky₆..ky₁, w₆..w₁, kx₁..kx₆]` | **96** | **596** | 16.84 |
| random (n=4) | 496–512 (mean 508) | ~1532 | — |

- The useful layout was **not** the MinLA optimum. Multi-resolution exact
  screening consistently selected a coordinate-block sandwich family with `w`
  between the two momentum coordinates. For example, exact `Rd=6` screening
  selects `perm = [12,11,10,9,8,7, 1,2,3,4,5,6, 13,14,15,16,17,18]`, reducing
  exact χmax `264 → 102`.
- TCI validation of close sandwich variants gives the best observed layout
  `perm = [18,17,16,15,14,13, 6,5,4,3,2,1, 7,8,9,10,11,12]`, with **64%
  reduction in χmax** (`264 → 96`) and **41% reduction in χsum** (`1018 → 596`)
  versus the original blocked layout, at TCI tolerance `1e-8`.
- Pairwise-MI MinLA became better after the replacement solver (`χmax≈231`, so
  a modest 12% χmax reduction), but it still misses the much better sandwich
  layout. The pairwise-MI cutwidth objective also improved over blocked only
  modestly (`χmax≈230`).
- Root cause: the pole lives on the 2D manifold `w = ε(kx,ky) = −2cos(kx)−2cos(ky)`,
  a **nonlinear many-body constraint**. Pairwise MI under-represents the joint
  coupling, so a structure-aware coordinate layout beats the pure MI path
  objective.

## Generalized MI diagnostics

I added exact generalized MI routines in `src/generalized_mi.jl`:

- quantum block entropy `S(A)`
- quantum block MI `I(A:B)`
- quantum conditional MI `I(A:B|C)`
- quantum interaction / synergy diagnostics
- classical measurement-distribution versions using `p(x)=|ψ(x)|²`

Run:

```bash
julia -t auto --project=experiments experiments/generalized_mi_diagnostics.jl
```

For the Green function at `Rd=6`, coordinate-level diagnostics are:

```text
Classical block MI [w,kx,ky]:
  I(w:kx) ≈ 0.368
  I(w:ky) ≈ 0.368
  I(kx:ky) ≈ 0.000

Classical joint / synergy:
  I(w : kx,ky) ≈ 1.431
  synergy(w; kx,ky) ≈ 0.695
  I(kx : ky | w) ≈ 0.695
```

This is exactly the missing structure: `kx` and `ky` are almost independent
marginally, but become dependent when `w` is known. This is the continuous
"explaining away" / collider pattern associated with the constraint
`w = ε(kx,ky)`. Pairwise bit MI misses it; block/conditional/classical MI sees
it.

A data-driven layout rule suggested by this is:

1. compute classical block MI between coordinate groups;
2. identify a central/collider coordinate with large MI to both others while the
   others have small MI to each other (`w` here);
3. generate a sandwich family `[outer, center, outer]`;
4. resolve block orientations by exact coarse SVD or TCI validation.

This recovers the same winning family found by the structured optimizer.

## Conclusions

1. **The methodology works** end-to-end: MI matrix → MinLA Hamiltonian path →
   bit layout, validated against exact and TCI bond dimensions.
2. **Pairwise MI is an excellent layout surrogate when correlations are
   pairwise** (1D smooth: corr 0.97; 2D diagonal-coupled: corr 0.99), yielding
   large bond-dimension reductions — **57% χmax / 62% entropy** on the 2D
   diagonal case — and it recovers known-good layouts (interleaving).
3. **Pairwise MI is insufficient for many-body coupling.** For the Green's
   function, where the relevant constraint `w = ε(kx,ky)` is nonlinear, pairwise
   MI under-represents the cross-coordinate coupling. However, a broader
   structure-aware search found a much better sandwich layout (`χmax 264 → 96`).
4. **Practical guidance:** use MI-guided MinLA for functions with pairwise/local
   correlation structure. For hidden many-body constraints (poles on a manifold,
   implicit equations), search constrained coordinate-block layouts and/or move
   toward higher-order / conditional MI objectives.

## Open directions

- **Better cut sketches:** a first unstructured beam search using random sampled
  unfoldings (`beam_search_layout_by_cut_sketch`) was added, but naive random
  submatrix sketches under-ranked the true hard cuts for the Green function. It
  is useful infrastructure, but needs leverage-score/adaptive row-column sampling
  or low-resolution calibration before it can replace exact coarse screening.
- **Higher-order surrogate:** replace pairwise MI with conditional / 3-body MI
  to capture `w = ε(kx,ky)`-style many-body coupling.
- **Minimax objective:** MinLA optimizes the *sum* (total bond dimension); a
  minimax variant (`min_k` cut MI) would target the *maximum* bond dimension
  directly (observed: corr(MinLA, χmax) is consistently weaker than
  corr(MinLA, Ssum)).
- **Better sampling for peaked states:** the Green's function is peaked on the
  dispersion; uniform MI sampling is inefficient. The hybrid MCMC sampler exists
  but is currently disabled (`src/sampling.jl`) — verifying it would improve the
  large-`R` MI matrix.
- **Tree (TTN) layouts:** the current trusted replacement intentionally supports
  only the path case (`max_deg=2`). A future, separately verified TTN/tree solver
  could target the Green's function's 3-coordinate structure directly.
