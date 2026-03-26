"""
render_targets.py — Render ground truth target images from a scene file.
=========================================================================
Produces a targets/ directory with:
  - view_0.npy, view_1.npy, ... — HDR float32 images (H, W, 3)
  - view_0.ppm, view_1.ppm, ... — tonemapped preview images
  - cameras.json — camera positions/targets for the optimizer to load
  - scene_info.json — metadata (resolution, num materials, etc.)

This runs ONCE, then the optimizer loads these files without re-rendering.

Usage:
  uv run render_targets.py --scene scenes/cornell_box.json
  uv run render_targets.py --scene scenes/cornell_box.json --spp 64 --output targets/cornell
  uv run render_targets.py --scene scenes/material_gallery.json --spp 128

The optimizer then runs:
  uv run train.py --load-targets targets/cornellbox --scene scenes/cornellbox.json
"""

import json
import os
import sys
import time

import numpy as np
import taichi as ti
import taichi.math as tm

# Import scene loader and rendering infrastructure
# We import the main diff_pathtracer to reuse its fields and render kernels
from diffpt.scene_loader import apply_ground_truth, load_scene

# Allocate minimal fields needed for rendering

MAX_QUADS = 16
MAX_SPHERES = 8
MAX_MATERIALS = 16
MAX_VIEWS = 8
MAX_DEPTH = 4
EPS = 1e-6
PI = 3.14159265358979
IMG_W = 256
IMG_H = 256

# # Runtime counts
# num_quads_field = None
# num_spheres_field = None
# num_materials_field = None
# num_views_field = None
# light_idx_field = None
# light_area_field = None

# # Materials
# albedo = None
# roughness = None
# metallic = None
# emission = None

# # Geometry
# quad_center = None
# quad_u = None
# quad_v = None
# quad_normal = None
# quad_mat = None
# quad_is_light = None
# sphere_center = None
# sphere_radius = None
# sphere_mat = None

# # Cameras
# cam_pos = None
# cam_fwd = None
# cam_right = None
# cam_up_vec = None
# current_view = None

# # Image
# image = None
# display = None
# rng_seed = None
# frame_count = None
# accumulator = None


def init_fields():
    global image, rng_seed, display, frame_count, accumulator
    global quad_center, quad_u, quad_v, quad_normal, quad_mat, quad_is_light
    global sphere_center, sphere_radius, sphere_mat
    global albedo, roughness, metallic, emission
    global num_quads_field, num_spheres_field, num_materials_field
    global light_idx_field, light_area_field
    global num_views_field, cam_pos, cam_fwd, cam_right, cam_up_vec, current_view

    image = ti.Vector.field(3, dtype=ti.f32, shape=(IMG_W, IMG_H))
    rng_seed = ti.field(dtype=ti.u32, shape=(IMG_W, IMG_H))
    display = ti.Vector.field(3, dtype=ti.f32, shape=(IMG_W, IMG_H))
    accumulator = ti.Vector.field(3, dtype=ti.f32, shape=(IMG_W, IMG_H))
    frame_count = ti.field(dtype=ti.i32, shape=())

    # Runtime counts
    num_quads_field = ti.field(dtype=ti.i32, shape=())
    num_spheres_field = ti.field(dtype=ti.i32, shape=())
    num_materials_field = ti.field(dtype=ti.i32, shape=())
    num_views_field = ti.field(dtype=ti.i32, shape=())
    light_idx_field = ti.field(dtype=ti.i32, shape=())
    light_area_field = ti.field(dtype=ti.f32, shape=())

    # Materials
    albedo = ti.Vector.field(3, dtype=ti.f32, shape=(MAX_MATERIALS,))
    roughness = ti.field(dtype=ti.f32, shape=(MAX_MATERIALS,))
    metallic = ti.field(dtype=ti.f32, shape=(MAX_MATERIALS,))
    emission = ti.Vector.field(3, dtype=ti.f32, shape=(1,))

    # Geometry
    quad_center = ti.Vector.field(3, dtype=ti.f32, shape=(MAX_QUADS,))
    quad_u = ti.Vector.field(3, dtype=ti.f32, shape=(MAX_QUADS,))
    quad_v = ti.Vector.field(3, dtype=ti.f32, shape=(MAX_QUADS,))
    quad_normal = ti.Vector.field(3, dtype=ti.f32, shape=(MAX_QUADS,))
    quad_mat = ti.field(dtype=ti.i32, shape=(MAX_QUADS,))
    quad_is_light = ti.field(dtype=ti.i32, shape=(MAX_QUADS,))
    sphere_center = ti.Vector.field(3, dtype=ti.f32, shape=(MAX_SPHERES,))
    sphere_radius = ti.field(dtype=ti.f32, shape=(MAX_SPHERES,))
    sphere_mat = ti.field(dtype=ti.i32, shape=(MAX_SPHERES,))

    # Cameras
    cam_pos = ti.Vector.field(3, dtype=ti.f32, shape=(MAX_VIEWS,))
    cam_fwd = ti.Vector.field(3, dtype=ti.f32, shape=(MAX_VIEWS,))
    cam_right = ti.Vector.field(3, dtype=ti.f32, shape=(MAX_VIEWS,))
    cam_up_vec = ti.Vector.field(3, dtype=ti.f32, shape=(MAX_VIEWS,))
    current_view = ti.field(dtype=ti.i32, shape=())


# GGX BRDF + intersection + path tracing (copied from main — needed for standalone)


@ti.func
def hash_u32(val: ti.u32) -> ti.u32:
    val = ti.atomic_xor(val, val >> 16)
    val = val * ti.u32(0x45D9F3B)
    val = ti.atomic_xor(val, val >> 16)
    val = val * ti.u32(0x45D9F3B)
    val = ti.atomic_xor(val, val >> 16)
    
    # Challenge 1
    return val


@ti.func
def next_rand(px: ti.i32, py: ti.i32) -> ti.f32:
    rng_seed[px, py] = hash_u32(rng_seed[px, py])  # update the seed
    result = ti.cast(rng_seed[px, py], ti.f32)  # convert to float
    result /= 4294967295.0

    # Challenge 1
    return result


@ti.func
def ggx_d(ndoth: ti.f32, alpha: ti.f32) -> ti.f32:
    result = ((ndoth * ndoth) * (alpha * alpha - 1.0) + 1.0)
    # Challenge 1
    return (alpha * alpha) / (PI * (result * result))


@ti.func
def smith_g1(ndotv: ti.f32, alpha: ti.f32) -> ti.f32:
    g1 = 2*ndotv / (ndotv + ti.sqrt(alpha * alpha + (1.0 - alpha * alpha) * (ndotv * ndotv)))

    # Challenge 1
    return g1


@ti.func
def fresnel_schlick(cos_theta: ti.f32, f0: tm.vec3) -> tm.vec3:
    fres = f0 + (1 - f0) * (1 - cos_theta) ** 5
    # Challenge 1
    return fres


"""
Before moving on to challenge section 2 run
uv run test.py --challenge-1
If it outputs success then move on otherwise review
solutions or ask for help
"""


@ti.func
def eval_brdf(
    wo: tm.vec3,
    wi: tm.vec3,
    n: tm.vec3,
    base_color: tm.vec3,
    rough: ti.f32,
    metal: ti.f32,
) -> tm.vec3:
    # Challenge 2
    ndot1 = max(ti.math.dot(n, wo), 0.0)
    ndotv = max(ti.math.dot(n, wo), EPS)
    h = ti.math.normalize(wi + wo)
    ndoth = max(ti.math.dot(n, h), 0.0)
    vdoth = max(ti.math.dot(wo, h), 0.0)
    alpha = max(rough * rough, 0.001)
    f0 = (1.0 - metal) * (0.04 * 0.04 * 0.04) + metal * base_color

    D = ggx_d(ndoth, alpha)
    F = fresnel_schlick(vdoth, f0)
    G = smith_g1(ndotv, alpha)

    spec = F * ((D * G) / (4.0 * ndotv * ndot1 + EPS))
    kd = (1.0 - F) * (1.0 - metal)
    diff= kd * base_color / PI

    return (diff + spec) * ndot1


@ti.func
def scene_intersect(ray_o: tm.vec3, ray_d: tm.vec3):
    nq = num_quads_field[None]
    ns = num_spheres_field[None]
    closest_t = ti.cast(1e10, ti.f32)
    hit_normal = tm.vec3(0.0, 1.0, 0.0)
    hit_mat = -1
    hit_light = 0

    # Challenge 2
    for quad_idx in range(MAX_QUADS):
        if quad_idx < nq:
            p = ray_o + closest_t * ray_d
            t = ti.math.dot(quad_center[quad_idx] - ray_o, quad_normal[quad_idx]) / ti.math.dot(ray_d, quad_normal[quad_idx])
            pu = ti.math.dot(p - quad_center[quad_idx], quad_u[quad_idx]) / ti.math.dot(quad_u[quad_idx], quad_u[quad_idx])
            pv = ti.math.dot(p - quad_center[quad_idx], quad_v[quad_idx]) / ti.math.dot(quad_v[quad_idx], quad_v[quad_idx])

            closest_t = ti.select((0.0 < pu < 1.0) and (0.0 < pv < 1.0) and t < closest_t, t, closest_t)
            hit_normal = ti.select((0.0 < pu < 1.0) and (0.0 < pv < 1.0) and t < closest_t, quad_normal[quad_idx], hit_normal)
            hit_mat = ti.select((0.0 < pu < 1.0) and (0.0 < pv < 1.0) and t < closest_t, quad_mat[quad_idx], hit_mat)
            hit_light = ti.select((0.0 < pu < 1.0) and (0.0 < pv < 1.0) and t < closest_t, quad_is_light[quad_idx], hit_light)

    for sphere_idx in range(MAX_SPHERES):
        if sphere_idx < ns:
            oc = ray_o - sphere_center[sphere_idx]
            a = ti.math.dot(ray_d, ray_d)
            b = 2.0 * ti.math.dot(oc, ray_d)
            c = ti.math.dot(oc, oc) - sphere_radius[sphere_idx] * sphere_radius[sphere_idx]
            discriminant = b * b - 4 * a * c

            t = ti.cast(1e10, ti.f32)
            hit_sphere = False
            if discriminant > 0:
                sqrt_disc = ti.sqrt(discriminant)
                t0 = (-b - sqrt_disc) / (2.0 * a)
                t1 = (-b + sqrt_disc) / (2.0 * a)
                t_candidate = ti.select((t0 > EPS) and (t0 < closest_t), t0, t1)
                hit_sphere = (t_candidate > EPS) and (t_candidate < closest_t)

            closest_t = ti.select(hit_sphere, t_candidate, closest_t)
            hit_normal = ti.select(hit_sphere, ti.math.normalize((ray_o + closest_t * ray_d) - sphere_center[sphere_idx]), hit_normal)
            hit_mat = ti.select(hit_sphere, sphere_mat[sphere_idx], hit_mat)
            hit_light = 0  # spheres are not lights in this scene
    
    if discriminant > 0 and hit_sphere:
        closest_t = t_candidate
        hit_normal = ti.math.normalize((ray_o + closest_t * ray_d) - sphere_center[sphere_idx])
        hit_mat = sphere_mat[sphere_idx]
        hit_light = 0

    return closest_t, hit_normal, hit_mat, hit_light


@ti.func
def visibility_test(p: tm.vec3, target: tm.vec3) -> ti.f32:
    d = ...
    dist = ...
    d_norm = ...
    nq = num_quads_field[None]
    ns = num_spheres_field[None]
    closest = ti.cast(1e10, ti.f32)

    # Challenge 2
    vis = 0.0
    if closest < dist - EPS:
        vis = 0.0
    else:
        vis = 1.0

    return vis


@ti.kernel
def render_high_spp(spp: ti.i32):
    # Challenge 3

    li = light_idx_field[None]
    la = light_area_field[None]
    view = current_view[None]
    for px, py in image:
        image[px, py] = ...


# Applies tone mapping and linearization for presentation
@ti.kernel
def tonemap():
    for i, j in display:
        n = ti.cast(frame_count[None], ti.f32)
        c = accumulator[i, j] / n
        c = c / (c + 1.0)
        display[i, j] = tm.pow(c, tm.vec3(1.0 / 2.2))


# Save utilities


def save_ppm(filename, arr):
    """Save numpy array (H, W, 3) float32 as PPM with gamma correction."""
    arr = np.clip(arr, 0.0, 1.0)
    arr = np.power(arr, 1.0 / 2.2)
    arr = (arr * 255).astype(np.uint8)
    h, w, _ = arr.shape
    with open(filename, "wb") as f:
        f.write(f"P6\n{w} {h}\n255\n".encode())
        f.write(arr.tobytes())


# Main


def main():
    scene_path = None
    output_dir = None
    render_spp = 64  # higher SPP for clean targets

    args = sys.argv[1:]
    Test_Render = False
    for i, arg in enumerate(args):
        if arg == "--scene" and i + 1 < len(args):
            scene_path = args[i + 1]
        elif arg == "--output" and i + 1 < len(args):
            output_dir = args[i + 1]
        elif arg == "--test-render":
            Test_Render = True
        elif arg == "--spp" and i + 1 < len(args):
            render_spp = int(args[i + 1])

    if scene_path is None:
        print(
            "Usage: uv run render_targets.py --scene scenes/cornell_box.json [--spp 64] [--output dir]"
        )
        sys.exit(1)

    # Read resolution from scene
    with open(scene_path) as f:
        scene_data = json.load(f)
        settings = scene_data.get("settings", {})
        IMG_W = settings.get("img_w", 256)
        IMG_H = settings.get("img_h", 256)

    if output_dir is None:
        scene_name = os.path.splitext(os.path.basename(scene_path))[0]
        output_dir = f"targets/{scene_name}"

    # Init Taichi
    ti.init(arch=ti.gpu)
    init_fields()

    _fields = {
        "quad_center": quad_center,
        "quad_u": quad_u,
        "quad_v": quad_v,
        "quad_normal": quad_normal,
        "quad_mat": quad_mat,
        "quad_is_light": quad_is_light,
        "sphere_center": sphere_center,
        "sphere_radius": sphere_radius,
        "sphere_mat": sphere_mat,
        "albedo": albedo,
        "roughness": roughness,
        "metallic": metallic,
        "emission": emission,
        "cam_pos": cam_pos,
        "cam_fwd": cam_fwd,
        "cam_right": cam_right,
        "cam_up_vec": cam_up_vec,
        "num_quads_field": num_quads_field,
        "num_spheres_field": num_spheres_field,
        "num_materials_field": num_materials_field,
        "num_views_field": num_views_field,
        "light_idx_field": light_idx_field,
        "light_area_field": light_area_field,
    }

    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading scene: {scene_path}")
    info = load_scene(scene_path, _fields)

    apply_ground_truth(info, _fields)

    print(f"Rendering {info.num_views} target views at SPP={render_spp}...")
    print(f"Output: {output_dir}/")
    print(f"Resolution: {IMG_W}x{IMG_H}")
    print("-" * 60)

    if Test_Render:
        window = ti.ui.Window("Path Tracer", (IMG_W, IMG_H))
        canvas = window.get_canvas()
        frame_count[None] = 0

        while window.running:
            image.fill(0.0)
            render_high_spp(1)

            frame_count[None] += 1

            @ti.kernel
            def add():
                for i, j in accumulator:
                    accumulator[i, j] += image[i, j]

            add()
            tonemap()
            canvas.set_image(display)
            window.show()
        sys.exit(0)

    start = time.time()
    camera_data = []

    for v in range(info.num_views):
        current_view[None] = v
        image.fill(0.0)
        render_high_spp(render_spp)

        arr = image.to_numpy()  # (W, H, 3)

        # Save HDR .npy (linear, float32 — this is what the optimizer loads)
        npy_path = os.path.join(output_dir, f"view_{v}.npy")
        np.save(npy_path, arr)

        # Save tonemapped preview
        ppm_path = os.path.join(output_dir, f"view_{v}.ppm")
        save_ppm(ppm_path, arr)

        elapsed = time.time() - start
        print(f"  View {v}: saved {npy_path} ({elapsed:.1f}s)")

        camera_data.append(
            {
                "position": info.cam_positions[v],
                "target": info.cam_targets[v],
            }
        )

    # Save camera metadata
    cam_path = os.path.join(output_dir, "cameras.json")
    with open(cam_path, "w") as f:
        json.dump({"cameras": camera_data}, f, indent=2)

    # Save scene info for the optimizer
    meta = {
        "scene_file": scene_path,
        "img_w": IMG_W,
        "img_h": IMG_H,
        "spp": render_spp,
        "num_views": info.num_views,
        "num_materials": info.num_materials,
        "material_names": info.material_names,
        "view_files": [f"view_{v}.npy" for v in range(info.num_views)],
    }
    meta_path = os.path.join(output_dir, "targets_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    total = time.time() - start
    print(f"\nDone in {total:.1f}s. Files saved to {output_dir}/")
    print(f"\nTo train, run:")
    print(f"  uv run train.py --load-targets {output_dir} --scene {scene_path}")


if __name__ == "__main__":
    main()
