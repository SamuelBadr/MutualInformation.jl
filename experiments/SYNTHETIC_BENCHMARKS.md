# Synthetic layout benchmarks

Purpose: create controlled black-box functions with known dependency motifs, then
rank layout candidates by **exact TT bond dimensions** (SVD ranks of unfoldings).
This is meant to test layout heuristics before applying them to expensive Hubbard
/ Parquet objects.

Run:

```bash
julia -t auto --project=experiments experiments/synthetic_layout_benchmarks.jl
```

## Benchmark motifs

| benchmark | motif | expected lesson |
|---|---|---|
| `separable_3coord` | product of one-coordinate factors | coordinate blocks are best; interleaving is harmful |
| `pairwise_diagonal_xy` | `x≈y` | scale interleaving wins |
| `collider_sum_z_eq_x_plus_y` | scalar additive constraint `z=x+y` | full scale interleaving wins, not block sandwich |
| `chain_x_y_z` | `x≈y≈z` chain | block layout with `y` middle is best among generated families, orientations matter |
| `ph_transfer_q_eq_kp_minus_k` | transfer relation `q=kp-k` | full scale interleaving wins |
| `star4_s_eq_x_plus_y_plus_z` | four-variable additive star | full scale interleaving wins |

## Headline results

Exact SVD ranks from one run:

| benchmark | natural χmax | best χmax | best family |
|---|---:|---:|---|
| separable | 4 | 4 | block |
| pairwise diagonal | 32 | 12 | scale interleave |
| scalar collider `z=x+y` | 64 | 16 | scale interleave |
| chain | 64 | 64 | block, better orientations reduce χsum |
| transfer `q=kp-k` | 64 | 16 | scale interleave |
| star4 additive | 34 | 17 | scale interleave |

## Key lesson

The word “collider” is not enough to determine the layout. There are at least two
important subclasses:

1. **Scalar arithmetic/additive constraints** like `z=x+y` or `q=kp-k`.
   These have carry/scale-local structure, so **scale interleaving** is often best.

2. **Coordinate-block dispersion constraints** like the 2D lattice Green's function
   `w = -2cos(kx)-2cos(ky)`. Here whole momentum coordinates are meaningful
   nonlinear blocks, and a **coordinate sandwich** can be much better.

This is why the layout optimizer must test both structured block motifs and scale
interleavings, and ultimately rank by exact/TCI bond dimensions rather than by a
single MI heuristic.
