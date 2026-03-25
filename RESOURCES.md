# References & Learning Resources

This document covers the theory behind every major feature implemented in Graph-A-Thon's differntial pathtracer, with links to papers, tutorials, and code references.

---

## 1. Monte Carlo Path Tracing

The foundation of the renderer. A path tracer estimates the rendering equation by tracing random light paths from the camera through the scene.

**The rendering equation** (Kajiya 1986) defines how light is transported in a scene. Our path tracer is a Monte Carlo estimator of this integral equation.

- **Kajiya, J.T.** (1986). *The Rendering Equation.* SIGGRAPH 1986. [ACM](https://dl.acm.org/doi/10.1145/15886.15902) — The foundational paper. Everything in our renderer flows from this.
- **Pharr, M., Jakob, W., Humphreys, G.** *Physically Based Rendering: From Theory to Implementation (PBRT).* [pbrt.org](https://www.pbrt.org/) — The definitive textbook. Chapters 13-14 cover Monte Carlo integration and path tracing. Free online.
- **Ray Tracing in One Weekend** by Peter Shirley — [raytracing.github.io](https://raytracing.github.io/) — Approachable introduction to building a ray tracer from scratch. Our initial path tracer follows a similar structure.

---

## 2. Next Event Estimation (NEE)

Also called "direct light sampling." At each surface hit, instead of hoping a random bounce direction hits the light, we explicitly sample a point on the light source and trace a shadow ray.

This is implemented in `trace_sample_nee()` in `kernels.py`, where `sample_light()` picks a uniform point on the area light quad, and `visibility_test()` traces the shadow ray.

- **Pharr et al.** PBRT Chapter 13.10 — *Direct Lighting.* Covers light sampling, shadow rays, and combining NEE with path continuation.
- **Veach, E.** (1997). *Robust Monte Carlo Methods for Light Transport Simulation.* PhD thesis, Stanford. [Stanford](https://graphics.stanford.edu/papers/veach_thesis/) — Chapter 9 covers direct lighting and multiple importance sampling. The gold standard reference.
- **Shirley, P. et al.** *Ray Tracing: The Rest of Your Life.* [raytracing.github.io](https://raytracing.github.io/books/RayTracingTheRestOfYourLife.html) — Chapter 6-7 cover light sampling and importance sampling in an accessible way.

---

## 3. GGX / Cook-Torrance Microfacet BRDF

Our shading model uses the Cook-Torrance microfacet BRDF with GGX (Trowbridge-Reitz) normal distribution, Smith height-correlated masking-shadowing, and Schlick Fresnel approximation. This is the same BRDF used in Unreal Engine 4/5, Frostbite, and most modern game engines.

The BRDF has three components:
- **D (Normal Distribution Function):** GGX/Trowbridge-Reitz — controls the shape of specular highlights
- **G (Geometric Shadowing-Masking):** Smith's method with GGX — models self-shadowing of microfacets
- **F (Fresnel):** Schlick approximation — controls reflectivity vs. view angle

These are implemented in `kernels.py` as `ggx_ndf()`, `smith_g1()`, `fresnel_schlick()`, and combined in `eval_brdf()`.

### Core Papers

- **Walter, B. et al.** (2007). *Microfacet Models for Refraction through Rough Surfaces.* EGSR 2007. [Cornell](https://www.cs.cornell.edu/~srm/publications/EGSR07-btdf.pdf) — Introduces the GGX distribution (named after "Ground Glass Unknown"). The foundational paper for the NDF we use.
- **Cook, R.L. and Torrance, K.E.** (1982). *A Reflectance Model for Computer Graphics.* ACM TOG. [ACM](https://dl.acm.org/doi/10.1145/357290.357293) — The original Cook-Torrance paper defining the microfacet BRDF framework `D*F*G / (4*NdotL*NdotV)`.
- **Trowbridge, T.S. and Reitz, K.P.** (1975). *Average Irregularity Representation of a Rough Surface for Ray Reflection.* JOSA. — The original 1975 paper deriving what we call "GGX." See Pharr's note on [why we should call it Trowbridge-Reitz](https://pharr.org/matt/blog/2022/05/06/trowbridge-reitz).
- **Schlick, C.** (1994). *An Inexpensive BRDF Model for Physically-based Rendering.* Computer Graphics Forum. — The Fresnel approximation `F0 + (1-F0)*(1-cos θ)^5` that we use.

### Tutorials & Practical Guides

- **LearnOpenGL — PBR Theory** — [learnopengl.com/PBR/Theory](https://learnopengl.com/PBR/Theory) — Excellent walkthrough with GLSL code for every component. The GGX NDF code there is nearly identical to our `ggx_ndf()`.
- **Brian Karis** (2013). *Specular BRDF Reference.* [graphicrants.blogspot.com](http://graphicrants.blogspot.com/2013/08/specular-brdf-reference.html) — Comprehensive reference of D, F, G function choices from a UE4 graphics programmer. Includes optimized HLSL.
- **Burley, B.** (2012). *Physically-Based Shading at Disney.* SIGGRAPH Course. [disney-animation.s3.amazonaws.com](https://media.disneyanimation.com/uploads/production/publication_asset/48/asset/s2012_pbs_disney_brdf_notes_v3.pdf) — Defines the Disney BRDF / metallic workflow that our material model is based on.
- **Karis, B.** (2013). *Real Shading in Unreal Engine 4.* SIGGRAPH Course. [blog.selfshadow.com](https://blog.selfshadow.com/publications/s2013-shading-course/) — UE4's choice of D=GGX, F=Schlick, G=Smith-GGX — the exact combination we implement.
- **Hoffman, N.** (2013). *Background: Physics and Math of Shading.* SIGGRAPH Course. [blog.selfshadow.com](https://blog.selfshadow.com/publications/s2013-shading-course/) — Deep background on microfacet theory, energy conservation, and Fresnel.
- **Coding Labs — Cook-Torrance** — [codinglabs.net](https://www.codinglabs.net/article_physically_based_rendering_cook_torrance.aspx) — Step-by-step implementation walkthrough.

---

## 4. Inverse Rendering / Differentiable Rendering

The core technique: optimize scene parameters by differentiating through the rendering process.

- **Kato, H. et al.** (2020). *Differentiable Rendering: A Survey.* [arXiv:2006.12057](https://arxiv.org/abs/2006.12057) — Comprehensive survey of the field.
- **Nimier-David, M. et al.** (2019). *Mitsuba 2: A Retargetable Forward and Inverse Renderer.* ACM TOG. [mitsuba-renderer.org](https://www.mitsuba-renderer.org/research/2019/retargetable/) — Mitsuba 2's architecture, which inspired our design.
- **Jakob, W. et al.** (2022). *Mitsuba 3.* [mitsuba.readthedocs.io](https://mitsuba.readthedocs.io/) — Mitsuba 3's Dr.Jit-based differentiable renderer. Our two-pass (trace then shade) architecture is a simplified version of their path-replay backpropagation.
- **Vicini, D. et al.** (2021). *Path Replay Backpropagation: Differentiating Light Paths using Constant Memory and Linear Time.* ACM TOG (SIGGRAPH). [rgl.epfl.ch](https://rgl.epfl.ch/publications/Vicini2021PathReplay) — The PRB technique that our gradient computation is based on: trace paths forward, store structure, replay only the differentiable shading on the backward pass.
- **Li, T.-M. et al.** (2018). *Differentiable Monte Carlo Ray Tracing through Edge Sampling.* ACM TOG (SIGGRAPH Asia). [people.csail.mit.edu/tzumao](https://people.csail.mit.edu/tzumao/diffrt/) — Addresses the silhouette discontinuity problem that makes geometry gradients hard.
- **Laine, S. et al.** (2020). *Modular Primitives for High-Performance Differentiable Rendering.* ACM TOG (SIGGRAPH Asia). [github.com/NVlabs/nvdiffrast](https://github.com/NVlabs/nvdiffrast) — Nvdiffrast, the GPU-rasterization-based differentiable renderer from NVIDIA. Different approach (rasterization vs. our ray tracing) but same goal.

---

## 5. Adam Optimizer

Used for geometry parameter optimization. Adam maintains per-parameter adaptive learning rates using exponential moving averages of the gradient (first moment) and squared gradient (second moment).

Implemented in `geo_optimizer.py` in `_adam_step_kernel()`.

- **Kingma, D.P. and Ba, J.** (2014). *Adam: A Method for Stochastic Optimization.* [arXiv:1412.6980](https://arxiv.org/abs/1412.6980) — The original Adam paper. Our implementation follows Algorithm 1 exactly: `m_t = β₁·m_{t-1} + (1-β₁)·g_t`, `v_t = β₂·v_{t-1} + (1-β₂)·g_t²`, bias correction, then `θ -= lr * m̂_t / (√v̂_t + ε)`.
- **Ruder, S.** (2016). *An Overview of Gradient Descent Optimization Algorithms.* [arXiv:1609.04747](https://arxiv.org/abs/1609.04747) — Clear comparison of SGD, momentum, Adam, and variants. Good for understanding why Adam is better than vanilla SGD for our noisy geometry gradients.

---

## 6. SPSA (Simultaneous Perturbation Stochastic Approximation)

Used for geometry gradients and roughness/metallic gradients where analytical differentiation through the rendering equation is not possible (visibility discontinuities, complex BRDF dependence).

SPSA estimates the full gradient from just 2 function evaluations regardless of parameter count, vs. 2N for standard finite differences. Implemented in `geo_optimizer.py` (`compute_spsa_gradients()`) and `train.py` (`compute_roughness_metallic_grads()`).

- **Spall, J.C.** (1992). *Multivariate Stochastic Approximation Using a Simultaneous Perturbation Gradient Approximation.* IEEE Trans. Automatic Control, 37(3). [jhuapl.edu](https://www.jhuapl.edu/spsa/pdf-spsa/spall_tac92.pdf) — The foundational SPSA paper.
- **Spall, J.C.** (1998). *An Overview of the Simultaneous Perturbation Method for Efficient Optimization.* Johns Hopkins APL Technical Digest, 19(4). [jhuapl.edu](https://www.jhuapl.edu/spsa/pdf-spsa/spall_an_overview.pdf) — Accessible overview with implementation guidelines and the 5-step algorithm we follow.
- **SPSA website** — [jhuapl.edu/SPSA](https://www.jhuapl.edu/SPSA/) — Maintained by Spall himself. Includes parameter tuning guides, code, and examples.
- **Wikipedia — SPSA** — [en.wikipedia.org](https://en.wikipedia.org/wiki/Simultaneous_perturbation_stochastic_approximation) — Good quick reference for the math.

---

## 7. Taichi Lang

The GPU computing framework everything is built on. Taichi compiles Python-decorated functions (`@ti.kernel`, `@ti.func`) into GPU or CPU machine code via LLVM/CUDA/Vulkan.

- **Hu, Y. et al.** (2019). *Taichi: A Language for High-Performance Computation on Spatially Sparse Data Structures.* ACM TOG (SIGGRAPH Asia). [taichi.graphics](https://taichi.graphics/) — The original Taichi paper.
- **Taichi documentation** — [docs.taichi-lang.org](https://docs.taichi-lang.org/) — Official docs. Key pages:
  - [Kernels and Functions](https://docs.taichi-lang.org/docs/kernel_function) — explains `@ti.kernel` vs `@ti.func`
  - [Fields](https://docs.taichi-lang.org/docs/field) — Taichi's data container (our SOA buffers)
  - [GGUI](https://docs.taichi-lang.org/docs/ggui) — the real-time visualization system we use
  - [Math Module](https://docs.taichi-lang.org/docs/math_module) — `taichi.math` GLSL-style functions
- **Taichi examples** — [github.com/taichi-dev/taichi](https://github.com/taichi-dev/taichi/tree/master/python/taichi/examples) — Official examples including path tracers and fluid sims.

---

## 8. Color Space & Image Handling

Our image loader converts between sRGB (what cameras/monitors use) and linear RGB (what the renderer computes in).

- **sRGB specification** — IEC 61966-2-1. The exact transfer function we implement in `image_loader.py` (`srgb_to_linear()`): `linear = ((srgb + 0.055) / 1.055)^2.4` for values above 0.04045.
- **Pharr et al.** PBRT Chapter 10.3 — *Color Spaces* — covers linear vs. sRGB and when to convert.

---

## 9. Russian Roulette Path Termination

Unbiased technique for terminating paths without introducing bias. A path continues with probability proportional to its throughput; if it survives, the throughput is divided by the survival probability to compensate.

- **Pharr et al.** PBRT Chapter 13.7 — *Russian Roulette and Splitting.*
- **Arvo, J. and Kirk, D.** (1990). *Particle Transport and Image Synthesis.* SIGGRAPH 1990. — Early application of Russian roulette to rendering.

---

## 10. Scene Description & Material Models

Our JSON scene format is inspired by Mitsuba's XML format but simplified. The metallic/roughness workflow follows the Disney/UE4 convention.

- **Burley, B.** (2012). *Physically-Based Shading at Disney.* — Defines the metallic/roughness parameterization we use: `F0 = lerp(0.04, base_color, metallic)`.
- **glTF 2.0 PBR specification** — [khronos.org/gltf](https://www.khronos.org/registry/glTF/specs/2.0/glTF-2.0.html#metallic-roughness-material) — The Khronos standard for metallic-roughness materials. Our material model matches this.

---

## Recommended Reading Order

If you're new to this area, here's a suggested path:

1. **Ray Tracing in One Weekend** — build intuition for ray-scene intersection and basic shading
2. **LearnOpenGL PBR Theory** — understand the GGX BRDF math with visual examples
3. **PBRT Chapters 13-14** — Monte Carlo integration and path tracing theory
4. **Karis 2013 (Specular BRDF Reference)** — see all the D/F/G options and why we picked these
5. **Vicini et al. 2021 (Path Replay Backpropagation)** — understand how we differentiate through the renderer
6. **Spall 1998 (SPSA Overview)** — the geometry gradient estimation technique
7. **Kingma & Ba 2014 (Adam)** — the optimizer for geometry parameters
