# Vid2Sim Differential Gaussian Rasterization

This is a modified version of the original [diff-gaussian-rasterization](https://github.com/graphdeco-inria/diff-gaussian-rasterization) repository to support rasterizing `depth`, `normal`, `extra` and `gs_w` attributes.

Here's an example of our Vid2Sim modified differential gaussian rasterization repo
```python
from diff_gauss import GaussianRasterizationSettings, GaussianRasterizer

gs_w = None
rendered_image, rendered_depth, rendered_norm, rendered_alpha, radii, extra, gs_w = rasterizer(
    means3D = means3D,
    means2D = means2D,
    shs = shs,
    colors_precomp = colors_precomp,
    opacities = opacity,
    scales = scales,
    rotations = rotations,
    cov3Ds_precomp = cov3D_precomp,
    extra_attrs = extra_attrs
)
```

## Adapted from the following repository
- [diff-gaussian-rasterization](https://github.com/slothfulxtx/diff-gaussian-rasterization)
- [original repository](https://github.com/graphdeco-inria/diff-gaussian-rasterization)
- [AbsGS Rasterizer](https://github.com/TY424/AbsGS/tree/main/submodules/diff-gaussian-rasterization-abs)

<section class="section" id="BibTeX">
  <div class="container is-max-desktop content">
    <h2 class="title">BibTeX</h2>
    <pre><code>@article{xie2024vid2sim,
  title={Vid2Sim: Realistic and Interactive Simulation from Video for Urban Navigation},
  author={Ziyang Xie and Zhizheng Liu and Zhenghao Peng and Wayne Wu and Bolei Zhou},
  journal={Preprint},
  year={2024}
}
</code></pre>
  </div>
</section>
