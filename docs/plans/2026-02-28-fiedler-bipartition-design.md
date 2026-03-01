# Fiedler Bipartition Dendrogram - Design

## Motivation

The existing dendrogram clusters **eigenvalues** by proximity (single-linkage on
1D scalars).  This reveals curvature scale hierarchy but ignores which
parameters participate in which modes.

Recursive spectral bipartition via the Fiedler vector clusters **parameters**
by their coupling structure in the Hessian.  Two parameters in the same cluster
have strong second-order interaction (perturbing one affects the gradient of the
other).  This is a more direct operationalization of "circuit" than eigenvalue
proximity.

No prior work appears to apply spectral bipartition of the Hessian coupling
graph for neural network circuit discovery.  Related work uses weight magnitudes
or gradient similarities for affinity matrices, not the full H_ij structure.

## Algorithm

Given Hessian H (P x P), build a dendrogram over parameters:

1. **Affinity matrix**: A_ij = |H_ij| for i != j, A_ii = 0
2. **Graph Laplacian**: L = D - A, where D = diag(row sums of A)
3. **Fiedler vector**: v_2 = eigenvector for 2nd-smallest eigenvalue lambda_2 of L
4. **Bipartition**: left = {i : v_2[i] > 0}, right = {i : v_2[i] <= 0}
5. **Recurse** on sub-Hessians H[left, left] and H[right, right]
6. **Merge height** = 1/lambda_2 (inverse algebraic connectivity)
7. Convert recursive tree to scipy linkage matrix Z (N-1, 4)

### Stopping criteria

- Partition has <= 1 parameter (leaf)
- lambda_2 is effectively zero (< 1e-12) - disconnected graph
- Recursion depth > 20 (safeguard; log2(5994) ~ 13)

### Merge height and monotonicity

Using 1/lambda_2 as merge height: tightly-coupled subgraphs have high lambda_2
(low merge height, near bottom of dendrogram), weakly-coupled groups have low
lambda_2 (high merge height, near top).

Monotonicity is NOT guaranteed by recursive bipartition.  When it fails (child
merge height > parent), we enforce it by clamping: child heights are set to
min(child_height, parent_height - epsilon).  This preserves tree topology while
making the dendrogram valid for scipy.

### Fiedler vector computation

For dense matrices up to ~6K: `scipy.linalg.eigh(L, subset_by_index=[0, 1])`
computes only the two smallest eigenvalues/vectors via LAPACK.  No need for
iterative solvers at this scale.

For sub-matrices during recursion, the same approach scales naturally (sub-
matrices get smaller at each level).

## Scope: MNIST experiment only

Target: LeNetTiny, 5994 parameters, 25 training checkpoints.

Full Hessians will be recomputed from saved model checkpoints and cached as
.npy files (~288 MB each, ~7 GB total) in results/hessians/.

The grokking experiment (95K params) is out of scope - it uses Lanczos and
doesn't have full Hessians.

## New files

### src/spectral_dendrogram.py

Core module with:
- `build_fiedler_dendrogram(H) -> Z` - main entry point, returns scipy linkage
- `_recursive_bipartition(H_sub, indices, depth) -> TreeNode` - recursive core
- `_tree_to_linkage(root, n) -> Z` - convert tree to scipy format
- `fiedler_split(H) -> (left_idx, right_idx, lambda2)` - single bipartition
- `parameter_layer_labels(model) -> labels` - map flat param index to layer name

### src/recompute_hessians.py

Standalone script:
- Loads model checkpoints from results/checkpoints/
- Computes full Hessian at each checkpoint
- Saves to results/hessians/step_NNNNNN.npy
- Skips checkpoints that already have cached Hessians

### src/run_fiedler.py

Pipeline script:
- Loads cached Hessians (fails if not present)
- Runs build_fiedler_dendrogram at each checkpoint
- Saves Fiedler linkage matrices to results/fiedler/
- Generates visualizations

### tests/test_spectral_dendrogram.py

- Known block-diagonal Hessian: verify Fiedler split separates blocks
- Single block: verify recursion terminates gracefully
- Monotonicity enforcement: verify clamped linkage matrix is valid
- Scipy linkage format: verify shape, valid indices, counts

## New visualizations (additions to src/visualize.py)

### 1. Fiedler dendrogram snapshots

Same layout as existing plot_dendrogram_snapshots but with leaves colored by
layer membership:
- conv1 (params 0-207): blue
- conv2 (params 208-3423): green
- classifier (params 3424-5993): red

Key question this answers: does the Fiedler bipartition respect layer
boundaries, or does it find cross-layer circuits?

### 2. Layer composition at cut levels

For a given dendrogram cut (e.g. k=5 clusters), show what fraction of each
cluster comes from each layer.  Repeat across training checkpoints.

This reveals when cross-layer circuits form or dissolve during training.

### 3. Algebraic connectivity over training

Plot lambda_2 (top-level) vs training step.  Sharp changes indicate
reorganization of the parameter coupling structure.

### 4. Side-by-side comparison

Eigenvalue dendrogram (existing) next to Fiedler dendrogram (new) at the same
checkpoint.  Visual comparison of two different views of the same Hessian.

## Data flow

```
results/checkpoints/step_NNNNNN.pt
        |
        v  (recompute_hessians.py)
results/hessians/step_NNNNNN.npy    [5994 x 5994 float64]
        |
        v  (run_fiedler.py)
results/fiedler/step_NNNNNN.npz     [linkage Z, lambda2, param_labels]
        |
        v  (visualize.py)
results/figures/fiedler_*.png
```
