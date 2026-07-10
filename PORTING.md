# Porting notes: momentsnerf -> momentstem

Source: `~/projects/momentsnerf` (PixelNeRF fork), specifically
`src/model/encoder.py` (`SpatialEncoder`), `GaborNet/GaborNet/GaborLayer.py`
(`GaborConv2d`), and `ZerNet/zernet/layers.py` (`ComplexZernike`).

## Ported faithfully

- **Gabor kernel formula** (`stem.gabor_kernel`): exact port of
  `GaborConv2d.calculate_weights`, including the `1e-3` sigma epsilon, the
  `1/(2*pi*sigma^2)` scaling, and the slightly asymmetric grid
  `linspace(-ceil(k/2)+1, ceil(k/2), k)`. Pinned by
  `tests/test_bank_regression.py::test_gabor_kernel_formula_reference`.
- **Gabor parameter scheme** (`stem.gabor_bank`): freq/theta/sigma/psi drawn
  exactly as in `GaborConv2d.__init__` (Meshgini et al. grid: 5 frequencies,
  8 orientations, sigma = pi/freq, psi ~ U[0, pi)).
- **Sum variant** = dense `GaborConv2d(3, 3)` (9 responses summed into 3
  channels), **concat variant** = grouped `GaborConv2d(3, 9, groups=3)`
  (channel `3i+o` = input `i`'s `o`-th response). The two variants share one
  kernel bank here, so sum is exactly the channel-fold of concat
  (`tests/test_stem_shapes.py::test_sum_is_channel_fold_of_concat_gabor`).
- **Zernike polynomial table** (`stem.zernike_polynomial`): verbatim from
  `ComplexZernikeFunction.forward`, j = 0..14, including its deviations from
  the standard OSA/Wikipedia table (j=4 defocus without the `2r^2-1`
  normalisation; j=7/8 coma written as `3r^3 sin/cos(3t)`; j=11/13 secondary
  astigmatism with `4r^4-3r^2`). We port the table the paper's code used
  rather than "correcting" it.

## Changed deliberately (and why)

1. **Filters are frozen buffers.** In `GaborConv2d` the Gabor parameters were
   `nn.Parameter(requires_grad=True)` and kernels were recomputed every
   forward. Whether they actually trained depended on what the optimizer was
   given. Here the bank is generated once from a committed seed
   (`GABOR_SEED=1234`), registered as a buffer, and pinned by fingerprint
   tests. `gabor-learn` is the explicit trainable control.
2. **Kernel size 11, `same` padding.** The NeRF `SpatialEncoder` used 5x5
   Gabor kernels with padding 0 or 5; `ImageEncoder` used 11x11. The BMVC
   claim and the project spec say 11x11; classification at 32x32 additionally
   requires spatial-size preservation, hence `padding = k//2` everywhere.
3. **Zernike reimplemented as fixed conv kernels.** This is the largest
   departure, and it is forced by a finding in the source:

   > **In the shipped momentsnerf encoder the Zernike stage was a no-op.**
   > `self.z0 = (ComplexZernike(j=j+1) for j in ...)` is a Python *generator*:
   > it is consumed on the first forward pass and empty forever after; its
   > `alpha` parameters are never registered as submodules, never reach the
   > optimizer, and stay at their init value 0 -- and `exp(i*0*F) = 1`
   > multiplies the image by one. Net effect: identity (plus a dtype cast).
   > Any measured MomentsNeRF gain was therefore attributable to the Gabor
   > stage (and possibly the dtype cast), not Zernike. This matters for how
   > we interpret H1-H3 and should be double-checked against the exact
   > commit/config used for the BMVC numbers.

   Since there is no working Zernike computation to port, we implement what
   the spec describes: the 15 polynomials evaluated on a pixel tiling of the
   unit disk (values outside rho > 1 zeroed), L2-normalised, applied as
   standard fixed 2D convolutions. In concat mode each Z_j filters the
   channel mean (15 output channels); in sum mode the mean Zernike kernel is
   applied depthwise so the 3-channel contract holds.
4. **Concat mode carries an identity passthrough** (RGB + 9 Gabor + 15
   Zernike = 27 channels). The original concat variant was Gabor-only
   (9 channels, no passthrough, conv1 re-widened by weight tiling). The
   passthrough guarantees the moment stem can never *lose* information
   relative to the vanilla baseline, which makes H1/H2 attributable to the
   added prior rather than to a changed input representation. Set
   `include_identity=False` to ablate.
5. **Kernel normalisation.** Gabor kernels keep their natural analytic scale
   (faithful). Zernike kernels are unit-L2 (new code, and the raw polynomial
   norms differ by orders of magnitude). The `random-fixed` control matches
   the per-kernel L2 norms of the moment bank it replaces, so it isolates
   filter *structure* under identical filter *energy*.
6. **No conv1 weight tiling.** The original concat variant tiled pretrained
   RGB conv1 weights to initialise the widened conv1. We train from scratch
   (`pretrained: false` in v1 configs), and with the CIFAR stem surgery
   conv1 is replaced anyway; the tiling trick is therefore not ported.
7. **Dropped entirely:** `index()`/latent grid-sampling, feature-pyramid
   upsampling, `latent_scaling`, custom `ConvEncoder`, and every other
   PixelNeRF coupling. The stem is a pure `nn.Module`: `(B,3,H,W) ->
   (B,out_channels,H,W)`.
