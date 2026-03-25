"""
diffpt/kernels.py — All Taichi kernels and functions in one compilation unit.

Taichi @ti.func are force-inlined at compile time and don't work reliably
across Python module boundaries in Taichi 1.7.x. Keeping all Taichi code
in a single file avoids cross-module @ti.func import issues while still
allowing clean separation of Python-level logic in other modules.

This file contains:
  - RNG (hash_u32, next_rand)
  - GGX BRDF (ggx_ndf, smith_g1, fresnel_schlick, eval_brdf)
  - Intersection (intersect_quad_t, intersect_sphere_t, scene_intersect, visibility_test)
  - NEE light sampling (sample_light)
  - Path tracing (trace_sample_nee)
  - Render kernels (render_kernel, render_fd_kernel, render_and_grad_kernel)
  - Loss kernels (compute_loss_view, compute_loss_fd_view)
  - Optimizer step kernels (sgd_step_materials)
  - Display kernel (compose_display)
  - Geometry step kernel (_apply_geo_adam_step)
"""

import taichi as ti
import taichi.math as tm

from .config import *
from .fields import *

# RNG


@ti.func
def hash_u32(val: ti.u32) -> ti.u32:

    return ...


@ti.func
def next_rand(px: ti.i32, py: ti.i32) -> ti.f32:

    return ...


# GGX Cook-Torrance BRDF


# same solution as GGX_D
@ti.func
def ggx_ndf(ndoth: ti.f32, alpha: ti.f32) -> ti.f32:

    return ...


@ti.func
def smith_g1(ndotv: ti.f32, alpha: ti.f32) -> ti.f32:

    return ...


@ti.func
def fresnel_schlick(cos_theta: ti.f32, f0: tm.vec3) -> tm.vec3:

    return ...


@ti.func
def eval_brdf(
    wo: tm.vec3,
    wi: tm.vec3,
    n: tm.vec3,
    base_color: tm.vec3,
    rough: ti.f32,
    metal: ti.f32,
) -> tm.vec3:
    ...
    return (diff + spec) * ndotl


# Intersection


@ti.func
def scene_intersect(ray_o: tm.vec3, ray_d: tm.vec3):
    ...
    return closest_t, hit_normal, hit_mat, hit_light


@ti.func
def visibility_test(p: tm.vec3, target: tm.vec3) -> ti.f32:
    ...
    return vis


# NEE Light Sampling


@ti.func
def sample_light(px: ti.i32, py: ti.i32):
    li = light_idx_field[None]
    u = next_rand(px, py) * 2.0 - 1.0
    v = next_rand(px, py) * 2.0 - 1.0
    p = quad_center[li] + quad_u[li] * u + quad_v[li] * v
    n = quad_normal[li]
    pdf = 1.0 / (light_area_field[None] + EPS)
    return p, n, pdf


# Path Tracing (single sample, NEE + GGX)


@ti.func
def trace_sample_nee(px: ti.i32, py: ti.i32, s: ti.i32, spp_count: ti.i32):
    """Trace one path with NEE + GGX BRDF.
    Returns: (radiance, bm0..bm5) where bm* are bounce material IDs.
    Now tracks 6 bounces for better indirect illumination gradient signal."""
    view = current_view[None]
    rng_seed[px, py] = hash_u32(ti.u32(px * 13 + py * 79 + s * 997 + view * 3571 + 1))
    jx = next_rand(px, py)
    jy = next_rand(px, py)
    u_coord = (ti.cast(px, ti.f32) + jx) / ti.cast(IMG_W, ti.f32) * 2.0 - 1.0
    v_coord = (ti.cast(py, ti.f32) + jy) / ti.cast(IMG_H, ti.f32) * 2.0 - 1.0
    ray_o = cam_pos[view]
    ray_d = tm.normalize(
        cam_fwd[view]
        + cam_right[view] * u_coord * 0.6
        + cam_up_vec[view] * v_coord * 0.6
    )
    throughput = tm.vec3(1.0)
    radiance = tm.vec3(0.0)
    alive = 1
    bm0 = -1
    bm1 = -1
    bm2 = -1
    bm3 = -1
    bm4 = -1
    bm5 = -1

    for bounce in range(MAX_DEPTH):
        if alive == 1:
            closest_t, hit_normal, hit_mat, hit_light = scene_intersect(ray_o, ray_d)
            if hit_mat < 0:
                alive = 0
            elif hit_light == 1:
                if bounce == 0:
                    radiance += throughput * emission[0]
                alive = 0
            else:
                hit_pos = ray_o + ray_d * closest_t
                n = hit_normal
                wo = -ray_d
                if bounce == 0:
                    bm0 = hit_mat
                if bounce == 1:
                    bm1 = hit_mat
                if bounce == 2:
                    bm2 = hit_mat
                if bounce == 3:
                    bm3 = hit_mat
                if bounce == 4:
                    bm4 = hit_mat
                if bounce == 5:
                    bm5 = hit_mat
                mat_a = albedo[hit_mat]
                mat_r = roughness[hit_mat]
                mat_m = metallic[hit_mat]

                # NEE
                light_p, light_n, light_pdf = sample_light(px, py)
                to_light = light_p - hit_pos
                light_dist = tm.length(to_light)
                wi_l = to_light / (light_dist + EPS)
                ndotl_l = tm.dot(n, wi_l)
                lcos = -tm.dot(light_n, wi_l)
                if ndotl_l > EPS and lcos > EPS:
                    vis = visibility_test(hit_pos + n * 1e-4, light_p)
                    if vis > 0.5:
                        brdf_v = eval_brdf(wo, wi_l, n, mat_a, mat_r, mat_m)
                        geom = lcos / (light_dist * light_dist + EPS)
                        radiance += (
                            throughput * brdf_v * emission[0] * geom / (light_pdf + EPS)
                        )

                # Continue path
                r1 = next_rand(px, py)
                r2 = next_rand(px, py)
                phi = 2.0 * PI * r1
                cos_theta = ti.sqrt(r2)
                sin_theta = ti.sqrt(1.0 - r2)
                up = tm.vec3(0.0, 1.0, 0.0)
                if ti.abs(n.y) > 0.99:
                    up = tm.vec3(1.0, 0.0, 0.0)
                t_vec = tm.normalize(tm.cross(up, n))
                b_vec = tm.cross(n, t_vec)
                wi_s = tm.normalize(
                    t_vec * (ti.cos(phi) * sin_theta)
                    + b_vec * (ti.sin(phi) * sin_theta)
                    + n * cos_theta
                )
                brdf_c = eval_brdf(wo, wi_s, n, mat_a, mat_r, mat_m)
                pdf_c = cos_theta / PI + EPS
                throughput *= brdf_c / pdf_c
                if bounce >= 4:
                    rr = ti.min(
                        ti.max(
                            ti.max(throughput.x, ti.max(throughput.y, throughput.z)),
                            0.05,
                        ),
                        0.95,
                    )
                    if next_rand(px, py) > rr:
                        alive = 0
                    else:
                        throughput /= rr
                if alive == 1:
                    ray_o = hit_pos + n * 1e-4
                    ray_d = wi_s

    return radiance, bm0, bm1, bm2, bm3, bm4, bm5


# Render Kernels


@ti.kernel
def render_kernel(spp_val: ti.i32):
    for px, py in image:
        col = tm.vec3(0.0)
        for s in range(spp_val):
            rad, _, _, _, _, _, _ = trace_sample_nee(px, py, s, spp_val)
            col += rad
        image[px, py] = col / ti.cast(spp_val, ti.f32)


@ti.kernel
def render_fd_kernel():
    for px, py in image_fd:
        col = tm.vec3(0.0)
        for s in range(SPP_FD):
            rad, _, _, _, _, _, _ = trace_sample_nee(px, py, s, SPP_FD)
            col += rad
        image_fd[px, py] = col / ti.cast(SPP_FD, ti.f32)


@ti.kernel
def render_and_grad_kernel():
    """Single-pass render + gradient accumulation.
    Tracks 6 bounces for better indirect illumination gradient signal
    (critical for wall albedo recovery via color bleeding)."""
    nv = num_views_field[None]
    v = current_view[None]
    for px, py in image:
        # Per-sample storage for radiance and bounce materials
        rad_0 = tm.vec3(0.0)
        rad_1 = tm.vec3(0.0)
        rad_2 = tm.vec3(0.0)
        rad_3 = tm.vec3(0.0)
        rad_4 = tm.vec3(0.0)
        rad_5 = tm.vec3(0.0)
        rad_6 = tm.vec3(0.0)
        rad_7 = tm.vec3(0.0)
        # 6 bounce materials per sample (vec6 as two vec3 of i32... use flat storage)
        bms_0 = ti.Vector([0, 0, 0, 0, 0, 0], dt=ti.i32)
        bms_1 = ti.Vector([0, 0, 0, 0, 0, 0], dt=ti.i32)
        bms_2 = ti.Vector([0, 0, 0, 0, 0, 0], dt=ti.i32)
        bms_3 = ti.Vector([0, 0, 0, 0, 0, 0], dt=ti.i32)
        bms_4 = ti.Vector([0, 0, 0, 0, 0, 0], dt=ti.i32)
        bms_5 = ti.Vector([0, 0, 0, 0, 0, 0], dt=ti.i32)
        bms_6 = ti.Vector([0, 0, 0, 0, 0, 0], dt=ti.i32)
        bms_7 = ti.Vector([0, 0, 0, 0, 0, 0], dt=ti.i32)

        col = tm.vec3(0.0)
        for s in range(SPP):
            rad, bm0, bm1, bm2, bm3, bm4, bm5 = trace_sample_nee(px, py, s, SPP)
            col += rad
            bm_vec = ti.Vector([bm0, bm1, bm2, bm3, bm4, bm5], dt=ti.i32)
            if s == 0:
                rad_0 = rad
                bms_0 = bm_vec
            if s == 1:
                rad_1 = rad
                bms_1 = bm_vec
            if s == 2:
                rad_2 = rad
                bms_2 = bm_vec
            if s == 3:
                rad_3 = rad
                bms_3 = bm_vec
            if s == 4:
                rad_4 = rad
                bms_4 = bm_vec
            if s == 5:
                rad_5 = rad
                bms_5 = bm_vec
            if s == 6:
                rad_6 = rad
                bms_6 = bm_vec
            if s == 7:
                rad_7 = rad
                bms_7 = bm_vec

        pixel_color = col / ti.cast(SPP, ti.f32)
        image[px, py] = pixel_color
        tgt = target_images[v, px, py]

        # Log-space loss gradient: d/dp [ (log(1+p) - log(1+t))^2 ]
        #   = 2 * (log(1+p) - log(1+t)) * 1/(1+p)
        # This compresses dynamic range so dark pixels (where wall color
        # bleeding shows) get proportionally stronger gradients.
        log_p = tm.vec3(
            ti.log(1.0 + ti.max(pixel_color.x, 0.0)),
            ti.log(1.0 + ti.max(pixel_color.y, 0.0)),
            ti.log(1.0 + ti.max(pixel_color.z, 0.0)),
        )
        log_t = tm.vec3(
            ti.log(1.0 + ti.max(tgt.x, 0.0)),
            ti.log(1.0 + ti.max(tgt.y, 0.0)),
            ti.log(1.0 + ti.max(tgt.z, 0.0)),
        )
        log_diff = log_p - log_t
        # Chain rule: d(log(1+p))/dp = 1/(1+p)
        inv_1p = tm.vec3(
            1.0 / (1.0 + ti.max(pixel_color.x, 0.0)),
            1.0 / (1.0 + ti.max(pixel_color.y, 0.0)),
            1.0 / (1.0 + ti.max(pixel_color.z, 0.0)),
        )
        dl_dp = 2.0 * log_diff * inv_1p / ti.cast(IMG_W * IMG_H * nv, ti.f32)
        inv_spp = 1.0 / ti.cast(SPP, ti.f32)

        for s in ti.static(range(SPP)):
            rad = tm.vec3(0.0)
            bm_vec = ti.Vector([0, 0, 0, 0, 0, 0], dt=ti.i32)
            if ti.static(s == 0):
                rad = rad_0
                bm_vec = bms_0
            if ti.static(s == 1):
                rad = rad_1
                bm_vec = bms_1
            if ti.static(s == 2):
                rad = rad_2
                bm_vec = bms_2
            if ti.static(s == 3):
                rad = rad_3
                bm_vec = bms_3
            if ti.static(s == 4):
                rad = rad_4
                bm_vec = bms_4
            if ti.static(s == 5):
                rad = rad_5
                bm_vec = bms_5
            if ti.static(s == 6):
                rad = rad_6
                bm_vec = bms_6
            if ti.static(s == 7):
                rad = rad_7
                bm_vec = bms_7

            if tm.length(rad) > EPS:
                em = emission[0]
                g_em = tm.vec3(0.0)
                if em.x > EPS:
                    g_em.x = rad.x / em.x
                if em.y > EPS:
                    g_em.y = rad.y / em.y
                if em.z > EPS:
                    g_em.z = rad.z / em.z
                ti.atomic_add(emission_grad[0], g_em * inv_spp * dl_dp)

                # Albedo gradient with bounce-depth weighting.
                # First-bounce material gets full weight, deeper bounces
                # get exponentially less. This prevents all materials from
                # receiving identical gradients when they share similar values.
                # Weight = 0.7^bounce: bounce 0→1.0, bounce 1→0.7, bounce 2→0.49, etc.
                for b in ti.static(range(MAX_DEPTH)):
                    bmat = bm_vec[b]
                    if bmat >= 0:
                        a = albedo[bmat]
                        g = tm.vec3(0.0)
                        if a.x > EPS:
                            g.x = rad.x / a.x
                        if a.y > EPS:
                            g.y = rad.y / a.y
                        if a.z > EPS:
                            g.z = rad.z / a.z
                        # Clamp gradient magnitude to prevent extreme values
                        # when albedo is very small
                        g = tm.clamp(g, -50.0, 50.0)
                        # Bounce decay: first-hit materials dominate
                        bounce_weight = 1.0
                        if ti.static(b == 1):
                            bounce_weight = 0.7
                        if ti.static(b == 2):
                            bounce_weight = 0.49
                        if ti.static(b == 3):
                            bounce_weight = 0.34
                        if ti.static(b == 4):
                            bounce_weight = 0.24
                        if ti.static(b == 5):
                            bounce_weight = 0.17
                        ti.atomic_add(
                            albedo_grad[bmat], g * inv_spp * dl_dp * bounce_weight
                        )


# Loss Kernels


@ti.kernel
def compute_loss_view(view: ti.i32):
    """Log-space L2 loss: compresses dynamic range so dark regions
    (where wall color bleeding is visible) contribute equally to
    bright regions (near the light). Uses log(1 + pixel) transform."""
    nv = num_views_field[None]
    for px, py in image:
        # Log-space transform: log(1 + x) compresses HDR range
        img = image[px, py]
        tgt = target_images[view, px, py]
        log_img = tm.vec3(
            ti.log(1.0 + ti.max(img.x, 0.0)),
            ti.log(1.0 + ti.max(img.y, 0.0)),
            ti.log(1.0 + ti.max(img.z, 0.0)),
        )
        log_tgt = tm.vec3(
            ti.log(1.0 + ti.max(tgt.x, 0.0)),
            ti.log(1.0 + ti.max(tgt.y, 0.0)),
            ti.log(1.0 + ti.max(tgt.z, 0.0)),
        )
        diff = log_img - log_tgt
        loss[None] += tm.dot(diff, diff) / ti.cast(IMG_W * IMG_H * nv, ti.f32)


@ti.kernel
def compute_loss_fd_view(view: ti.i32):
    """Log-space L2 loss for FD probes."""
    nv = num_views_field[None]
    for px, py in image_fd:
        img = image_fd[px, py]
        tgt = target_images[view, px, py]
        log_img = tm.vec3(
            ti.log(1.0 + ti.max(img.x, 0.0)),
            ti.log(1.0 + ti.max(img.y, 0.0)),
            ti.log(1.0 + ti.max(img.z, 0.0)),
        )
        log_tgt = tm.vec3(
            ti.log(1.0 + ti.max(tgt.x, 0.0)),
            ti.log(1.0 + ti.max(tgt.y, 0.0)),
            ti.log(1.0 + ti.max(tgt.z, 0.0)),
        )
        diff = log_img - log_tgt
        loss_fd[None] += tm.dot(diff, diff) / ti.cast(IMG_W * IMG_H * nv, ti.f32)


# Adam Optimizer Kernels
# β1=0.9, β2=0.999, ε=1e-8 (standard Adam defaults)


@ti.kernel
def adam_step_albedo(
    lr: ti.f32, beta1: ti.f32, beta2: ti.f32, adam_eps: ti.f32, t: ti.i32
):
    """Adam step for albedo."""
    nm = num_materials_field[None]
    bc1 = 1.0 - ti.pow(beta1, ti.cast(t, ti.f32))
    bc2 = 1.0 - ti.pow(beta2, ti.cast(t, ti.f32))
    for i in range(MAX_MATERIALS):
        if i < nm:
            g = albedo_grad[i]
            albedo_adam_m[i] = beta1 * albedo_adam_m[i] + (1.0 - beta1) * g
            albedo_adam_v[i] = beta2 * albedo_adam_v[i] + (1.0 - beta2) * g * g
            m_hat = albedo_adam_m[i] / bc1
            v_hat = albedo_adam_v[i] / bc2
            step = tm.vec3(
                m_hat.x / (ti.sqrt(v_hat.x) + adam_eps),
                m_hat.y / (ti.sqrt(v_hat.y) + adam_eps),
                m_hat.z / (ti.sqrt(v_hat.z) + adam_eps),
            )
            albedo[i] -= lr * step
            albedo[i] = tm.clamp(albedo[i], 0.01, 0.99)


@ti.kernel
def adam_step_emission(
    lr: ti.f32, beta1: ti.f32, beta2: ti.f32, adam_eps: ti.f32, t: ti.i32
):
    """Adam step for emission."""
    bc1 = 1.0 - ti.pow(beta1, ti.cast(t, ti.f32))
    bc2 = 1.0 - ti.pow(beta2, ti.cast(t, ti.f32))
    for i in range(1):
        g = emission_grad[i]
        emission_adam_m[i] = beta1 * emission_adam_m[i] + (1.0 - beta1) * g
        emission_adam_v[i] = beta2 * emission_adam_v[i] + (1.0 - beta2) * g * g
        m_hat = emission_adam_m[i] / bc1
        v_hat = emission_adam_v[i] / bc2
        step = tm.vec3(
            m_hat.x / (ti.sqrt(v_hat.x) + adam_eps),
            m_hat.y / (ti.sqrt(v_hat.y) + adam_eps),
            m_hat.z / (ti.sqrt(v_hat.z) + adam_eps),
        )
        emission[i] -= lr * step
        emission[i] = tm.clamp(emission[i], 1.0, 50.0)


@ti.kernel
def adam_step_roughness_metallic(
    lr: ti.f32,
    beta1: ti.f32,
    beta2: ti.f32,
    adam_eps: ti.f32,
    t: ti.i32,
    grad_clip: ti.f32,
):
    """Adam step for roughness and metallic with gradient clipping."""
    nm = num_materials_field[None]
    bc1 = 1.0 - ti.pow(beta1, ti.cast(t, ti.f32))
    bc2 = 1.0 - ti.pow(beta2, ti.cast(t, ti.f32))
    for i in range(MAX_MATERIALS):
        if i < nm:
            # Roughness
            rg = tm.clamp(roughness_grad[i], -grad_clip, grad_clip)
            roughness_adam_m[i] = beta1 * roughness_adam_m[i] + (1.0 - beta1) * rg
            roughness_adam_v[i] = beta2 * roughness_adam_v[i] + (1.0 - beta2) * rg * rg
            rm_hat = roughness_adam_m[i] / bc1
            rv_hat = roughness_adam_v[i] / bc2
            roughness[i] -= lr * rm_hat / (ti.sqrt(rv_hat) + adam_eps)
            roughness[i] = tm.clamp(roughness[i], 0.05, 1.0)

            # Metallic
            mg = tm.clamp(metallic_grad[i], -grad_clip, grad_clip)
            metallic_adam_m[i] = beta1 * metallic_adam_m[i] + (1.0 - beta1) * mg
            metallic_adam_v[i] = beta2 * metallic_adam_v[i] + (1.0 - beta2) * mg * mg
            mm_hat = metallic_adam_m[i] / bc1
            mv_hat = metallic_adam_v[i] / bc2
            metallic[i] -= lr * mm_hat / (ti.sqrt(mv_hat) + adam_eps)
            metallic[i] = tm.clamp(metallic[i], 0.0, 1.0)


# Display Kernel


@ti.kernel
def compose_display(view: ti.i32):
    for x, y in display_image:
        col = tm.vec3(0.0)
        if x < IMG_W:
            col = target_images[view, x, y]
        else:
            col = image[x - IMG_W, y]
        col = tm.clamp(col, 0.0, 1.0)
        col.x = ti.pow(col.x, 1.0 / 2.2)
        col.y = ti.pow(col.y, 1.0 / 2.2)
        col.z = ti.pow(col.z, 1.0 / 2.2)
        display_image[x, y] = col


# Geometry Step Kernel (used by geo_optimizer via train.py)


@ti.kernel
def apply_geo_adam_step(grad_center: ti.template(), grad_radius: ti.template()):
    """Apply Adam step stored in external gradient fields to sphere geometry."""
    ns = num_spheres_field[None]
    for i in range(MAX_SPHERES):
        if i < ns:
            sphere_center[i] -= grad_center[i]
            sphere_center[i] = tm.clamp(
                sphere_center[i], tm.vec3(-2.0, 0.1, -2.0), tm.vec3(2.0, 2.0, 2.0)
            )
            sphere_radius[i] -= grad_radius[i]
            sphere_radius[i] = tm.clamp(sphere_radius[i], 0.05, 0.8)


# Helper: render dispatch


def do_render(spp_val, view=None):
    """High-level render: set view, clear image, render."""
    if view is not None:
        current_view[None] = view
    image.fill(0.0)
    render_kernel(spp_val)
