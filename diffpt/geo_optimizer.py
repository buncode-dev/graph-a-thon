"""
geo_optimizer.py — Improved geometry gradient estimation for sphere primitives.

Three techniques combined:

1. Analytical Interior Gradients
   For rays that hit a sphere interior (not near silhouette), the intersection
   point and normal are smooth functions of sphere center/radius:
     hit_pos(c, r) = ray_o + ray_d * t(c, r)
     normal(c, r) = normalize(hit_pos - c)
   where t is the smooth root of the quadratic. We can differentiate the
   shading contribution (BRDF * throughput) w.r.t. the hit point/normal,
   then chain-rule through dt/dc and dt/dr. This gives exact gradients
   for interior pixels at zero extra render cost.

2. SPSA (Simultaneous Perturbation Stochastic Approximation)
   For silhouette pixels (where the smooth gradient is zero but the true
   gradient is nonzero due to visibility changes), we use SPSA:
     g ≈ (L(θ + c*delta) - L(θ - c*delta)) / (2c) * delta^{-1}
   where delta is a random ±1 vector. This estimates the full gradient from
   just 2 renders (vs 2N for standard FD). Noisier per-iteration but
   vastly cheaper, especially as sphere count grows.

3. ADAM OPTIMIZER
   Adam's per-parameter adaptive learning rate handles the very different gradient magnitudes
   between center xyz (large) and radius (small), and between analytical
   interior gradients (smooth, consistent) and SPSA silhouette gradients
   (noisy, sporadic).

Usage:
    from geo_optimizer import GeoOptimizer
    geo_opt = GeoOptimizer(num_spheres, fields_dict)

    # Each iteration:
    geo_opt.compute_analytical_gradients(...)
    geo_opt.compute_spsa_gradients(...)
    geo_opt.step(lr, iteration)
"""

import numpy as np
import taichi as ti
import taichi.math as tm

MAX_SPHERES = 8
EPS = 1e-6
PI = 3.14159265358979


# Additional fields for geometry optimizer


# Analytical gradient accumulators (filled during main render)
sphere_center_grad_analytical = ti.Vector.field(3, dtype=ti.f32, shape=(MAX_SPHERES,))
sphere_radius_grad_analytical = ti.field(dtype=ti.f32, shape=(MAX_SPHERES,))

# SPSA gradient accumulators
sphere_center_grad_spsa = ti.Vector.field(3, dtype=ti.f32, shape=(MAX_SPHERES,))
sphere_radius_grad_spsa = ti.field(dtype=ti.f32, shape=(MAX_SPHERES,))

# Combined gradient
sphere_center_grad_combined = ti.Vector.field(3, dtype=ti.f32, shape=(MAX_SPHERES,))
sphere_radius_grad_combined = ti.field(dtype=ti.f32, shape=(MAX_SPHERES,))

# Adam state
adam_m_center = ti.Vector.field(3, dtype=ti.f32, shape=(MAX_SPHERES,))  # first moment
adam_v_center = ti.Vector.field(3, dtype=ti.f32, shape=(MAX_SPHERES,))  # second moment
adam_m_radius = ti.field(dtype=ti.f32, shape=(MAX_SPHERES,))
adam_v_radius = ti.field(dtype=ti.f32, shape=(MAX_SPHERES,))

# Per-pixel sphere intersection info for analytical gradients
# hit_sphere_idx[px,py] = which sphere was hit at bounce 0 (-1 if none)
# hit_sphere_disc[px,py] = discriminant value (for silhouette detection)
hit_sphere_idx = None  # allocated dynamically based on resolution
hit_sphere_disc = None


def allocate_geo_fields(img_w, img_h):
    """Must be called before any kernels that use these fields."""
    global hit_sphere_idx, hit_sphere_disc
    hit_sphere_idx = ti.field(dtype=ti.i32, shape=(img_w, img_h))
    hit_sphere_disc = ti.field(dtype=ti.f32, shape=(img_w, img_h))


class GeoOptimizer:
    """Manages geometry optimization with analytical + SPSA gradients + Adam."""

    def __init__(self, num_spheres: int, fields: dict, img_w=256, img_h=256):
        self.num_spheres = num_spheres
        self.fields = fields
        self.img_w = img_w
        self.img_h = img_h

        # SPSA parameters
        self.spsa_c = 0.01  # perturbation magnitude
        self.spsa_c_decay = 0.999
        self.spsa_min_c = 0.002

        # Adam parameters
        self.adam_beta1 = 0.9
        self.adam_beta2 = 0.999
        self.adam_eps = 1e-8

        # Blending weight between analytical and SPSA
        self.analytical_weight = 0.7
        self.spsa_weight = 0.3

        # Allocate per-pixel fields
        allocate_geo_fields(img_w, img_h)

        self.reset()

    def reset(self):
        """Reset optimizer state."""
        adam_m_center.fill(0.0)
        adam_v_center.fill(0.0)
        adam_m_radius.fill(0.0)
        adam_v_radius.fill(0.0)
        sphere_center_grad_analytical.fill(0.0)
        sphere_radius_grad_analytical.fill(0.0)
        sphere_center_grad_spsa.fill(0.0)
        sphere_radius_grad_spsa.fill(0.0)
        sphere_center_grad_combined.fill(0.0)
        sphere_radius_grad_combined.fill(0.0)
        self.spsa_c = 0.01

    def zero_analytical_grads(self):
        sphere_center_grad_analytical.fill(0.0)
        sphere_radius_grad_analytical.fill(0.0)

    def compute_spsa_gradients(
        self, info, render_fd_fn, compute_loss_fn, current_view_field
    ):
        """SPSA: perturb all geo params simultaneously, estimate gradient from 2 renders.

        Args:
            info: SceneInfo object
            render_fd_fn: callable() that renders into image_fd at low SPP
            compute_loss_fn: callable(view) that computes loss from image_fd vs target
            current_view_field: Taichi field for current view index
        """
        sphere_center_grad_spsa.fill(0.0)
        sphere_radius_grad_spsa.fill(0.0)

        ns = self.num_spheres
        # Generate random perturbation direction: ±1 for each parameter
        # 3 center coords + 1 radius per sphere
        num_params = ns * 4
        delta = np.random.choice([-1.0, 1.0], size=num_params).astype(np.float32)
        c = self.spsa_c

        # Save original params
        orig_centers = []
        orig_radii = []
        for si in range(ns):
            orig_centers.append(
                [
                    self.fields["sphere_center"][si][0],
                    self.fields["sphere_center"][si][1],
                    self.fields["sphere_center"][si][2],
                ]
            )
            orig_radii.append(self.fields["sphere_radius"][si])

        # Positive perturbation
        for si in range(ns):
            base = si * 4
            self.fields["sphere_center"][si] = [
                orig_centers[si][0] + c * delta[base + 0],
                orig_centers[si][1] + c * delta[base + 1],
                orig_centers[si][2] + c * delta[base + 2],
            ]
            self.fields["sphere_radius"][si] = orig_radii[si] + c * delta[base + 3]

        loss_plus = 0.0
        for v in range(info.num_views):
            current_view_field[None] = v
            render_fd_fn()
            loss_plus += compute_loss_fn(v)

        # Negative perturbation
        for si in range(ns):
            base = si * 4
            self.fields["sphere_center"][si] = [
                orig_centers[si][0] - c * delta[base + 0],
                orig_centers[si][1] - c * delta[base + 1],
                orig_centers[si][2] - c * delta[base + 2],
            ]
            self.fields["sphere_radius"][si] = orig_radii[si] - c * delta[base + 3]

        loss_minus = 0.0
        for v in range(info.num_views):
            current_view_field[None] = v
            render_fd_fn()
            loss_minus += compute_loss_fn(v)

        # Restore original
        for si in range(ns):
            self.fields["sphere_center"][si] = orig_centers[si]
            self.fields["sphere_radius"][si] = orig_radii[si]

        # SPSA gradient estimate: g_i = (L+ - L-) / (2*c*delta_i)
        scalar_diff = (loss_plus - loss_minus) / (2.0 * c)
        for si in range(ns):
            base = si * 4
            gc = [scalar_diff / delta[base + j] for j in range(3)]
            sphere_center_grad_spsa[si] = gc
            sphere_radius_grad_spsa[si] = scalar_diff / delta[base + 3]

        # Decay perturbation magnitude
        self.spsa_c = max(self.spsa_c * self.spsa_c_decay, self.spsa_min_c)

    def combine_gradients(self):
        """Blend analytical (interior) and SPSA (silhouette) gradients."""
        _combine_grads_kernel(
            self.num_spheres, self.analytical_weight, self.spsa_weight
        )

    def step(self, lr: float, iteration: int):
        """Adam optimizer step for geometry parameters."""
        self.combine_gradients()
        t = iteration + 1
        _adam_step_kernel(
            self.num_spheres, lr, self.adam_beta1, self.adam_beta2, self.adam_eps, t
        )


@ti.kernel
def _combine_grads_kernel(ns: ti.i32, w_analytical: ti.f32, w_spsa: ti.f32):
    for i in range(MAX_SPHERES):
        if i < ns:
            sphere_center_grad_combined[i] = (
                w_analytical * sphere_center_grad_analytical[i]
                + w_spsa * sphere_center_grad_spsa[i]
            )
            sphere_radius_grad_combined[i] = (
                w_analytical * sphere_radius_grad_analytical[i]
                + w_spsa * sphere_radius_grad_spsa[i]
            )


@ti.kernel
def _adam_step_kernel(
    ns: ti.i32, lr: ti.f32, beta1: ti.f32, beta2: ti.f32, eps: ti.f32, t: ti.i32
):
    # Challenge 7
    """Adam update for sphere center and radius."""

    ...


# Analytical gradient helpers (called from the main render kernel)

# These functions compute d(hit_pos)/d(sphere_center) and d(normal)/d(sphere_center)
# for sphere intersections, allowing the main render kernel to accumulate
# analytical geometry gradients alongside material gradients.


@ti.func
def sphere_intersection_derivs(
    ray_o: tm.vec3, ray_d: tm.vec3, center: tm.vec3, radius: ti.f32
):
    """Compute intersection t and its derivatives w.r.t. center and radius.

    Returns: t, dt_dcx, dt_dcy, dt_dcz, dt_dr, discriminant
    All zero if no intersection.
    """
    oc = ray_o - center
    a = tm.dot(ray_d, ray_d)
    b = 2.0 * tm.dot(oc, ray_d)
    c_val = tm.dot(oc, oc) - radius * radius
    disc = b * b - 4.0 * a * c_val

    t = 0.0
    dt_dcx = 0.0
    dt_dcy = 0.0
    dt_dcz = 0.0
    dt_dr = 0.0

    if disc > EPS:
        sq = ti.sqrt(disc)
        t = (-b - sq) / (2.0 * a)
        if t < EPS:
            t = (-b + sq) / (2.0 * a)

        if t > EPS:
            # dt/d(center_k) = -d(discriminant root)/d(center_k)
            # b = 2 * dot(oc, ray_d), db/dc_k = -2 * ray_d_k
            # c = dot(oc,oc) - r², dc/dc_k = -2 * oc_k
            # disc = b² - 4ac
            # d(disc)/dc_k = 2*b*(-2*ray_d_k) - 4*a*(-2*oc_k)
            #              = -4*ray_d_k*b + 8*a*oc_k
            # t = (-b - sqrt(disc)) / (2a)
            # dt/dc_k = (-(-2*ray_d_k) - d(disc)/(2*sqrt(disc))) / (2a)
            #         = (2*ray_d_k - d_disc_k / (2*sq)) / (2a)

            d_disc_dcx = -4.0 * ray_d.x * b + 8.0 * a * oc.x
            d_disc_dcy = -4.0 * ray_d.y * b + 8.0 * a * oc.y
            d_disc_dcz = -4.0 * ray_d.z * b + 8.0 * a * oc.z

            inv_2a = 1.0 / (2.0 * a)
            inv_2sq = 1.0 / (2.0 * sq + EPS)

            dt_dcx = (2.0 * ray_d.x - d_disc_dcx * inv_2sq) * inv_2a
            dt_dcy = (2.0 * ray_d.y - d_disc_dcy * inv_2sq) * inv_2a
            dt_dcz = (2.0 * ray_d.z - d_disc_dcz * inv_2sq) * inv_2a

            # dt/dr: disc depends on r through c_val = dot(oc,oc) - r²
            # d(disc)/dr = -4*a*(-2*r) = 8*a*r
            d_disc_dr = 8.0 * a * radius
            dt_dr = -d_disc_dr * inv_2sq * inv_2a

    return t, dt_dcx, dt_dcy, dt_dcz, dt_dr, disc


@ti.func
def accumulate_geo_grad_for_sphere(
    px: ti.i32,
    py: ti.i32,
    sphere_idx: ti.i32,
    ray_o: tm.vec3,
    ray_d: tm.vec3,
    throughput: tm.vec3,
    emission_val: tm.vec3,
    dl_dp: tm.vec3,
    inv_spp: ti.f32,
    center: tm.vec3,
    radius: ti.f32,
    brdf_val: tm.vec3,
    ndotl: ti.f32,
    wo: tm.vec3,
    wi: tm.vec3,
    n: tm.vec3,
):
    """Accumulate analytical gradient for sphere geometry.

    This computes the chain: dL/d(geo) = dL/d(pixel) * d(pixel)/d(radiance) * d(radiance)/d(geo)

    For a diffuse/glossy surface lit by NEE:
      radiance = throughput * brdf(wo, wi, n(geo)) * emission * geom_term
    The main dependence on geometry is through the normal n, which changes
    the BRDF evaluation and thus the radiance.

    For simplicity and robustness, we compute:
      d(radiance)/d(geo) ≈ radiance * d(n)/d(geo) projected appropriately

    This is an approximation but captures the dominant signal for interior pixels.
    """
    t, dt_dcx, dt_dcy, dt_dcz, dt_dr, disc = sphere_intersection_derivs(
        ray_o, ray_d, center, radius
    )

    if t > EPS and disc > 0.01:  # only for non-silhouette (disc well above 0)
        hit_pos = ray_o + ray_d * t
        raw_n = hit_pos - center
        n_len = tm.length(raw_n)

        if n_len > EPS:
            # d(hit_pos)/d(cx) = ray_d * dt/dcx, etc.
            dhit_dcx = ray_d * dt_dcx
            dhit_dcy = ray_d * dt_dcy
            dhit_dcz = ray_d * dt_dcz
            dhit_dr = ray_d * dt_dr

            # d(raw_n)/d(cx) = d(hit_pos)/d(cx) - [1,0,0]
            draw_dcx = dhit_dcx - tm.vec3(1.0, 0.0, 0.0)
            draw_dcy = dhit_dcy - tm.vec3(0.0, 1.0, 0.0)
            draw_dcz = dhit_dcz - tm.vec3(0.0, 0.0, 1.0)
            draw_dr = dhit_dr  # center doesn't change, only hit_pos

            # d(normalized_n)/d(param) is complex (involves Jacobian of normalization)
            # Simplified: the dominant effect is n changes direction, which rotates
            # the BRDF lobe. We approximate d(radiance)/d(n) ≈ |radiance| * |dn|
            # with sign from dot(dn, light_dir).
            #
            # More precisely, for Lambertian: rad ∝ dot(n, wi),
            # so d(rad)/d(n_k) ∝ wi_k, giving d(rad)/d(param) = wi . d(n)/d(param)

            # Use the chain: d(rad)/d(param) ≈ [radiance_magnitude] * dot(wi, dn/d_param) / dot(n, wi)
            rad_mag = tm.length(throughput * emission_val) + EPS
            ndotwi = ti.max(tm.dot(n, wi), EPS)

            # dn/d(param) ≈ d(raw_n)/d(param) / n_len  (first-order)
            inv_nlen = 1.0 / n_len
            dn_dcx = draw_dcx * inv_nlen
            dn_dcy = draw_dcy * inv_nlen
            dn_dcz = draw_dcz * inv_nlen
            dn_dr = draw_dr * inv_nlen

            # Scalar gradient contribution per param
            grad_cx = rad_mag * tm.dot(wi, dn_dcx) / ndotwi
            grad_cy = rad_mag * tm.dot(wi, dn_dcy) / ndotwi
            grad_cz = rad_mag * tm.dot(wi, dn_dcz) / ndotwi
            grad_r = rad_mag * tm.dot(wi, dn_dr) / ndotwi

            # Scale by loss gradient and inv_spp
            scale = (dl_dp.x + dl_dp.y + dl_dp.z) / 3.0 * inv_spp

            ti.atomic_add(
                sphere_center_grad_analytical[sphere_idx],
                tm.vec3(grad_cx, grad_cy, grad_cz) * scale,
            )
            ti.atomic_add(sphere_radius_grad_analytical[sphere_idx], grad_r * scale)
