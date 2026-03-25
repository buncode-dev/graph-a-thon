#!/usr/bin/env python3
"""
train.py — Inverse Rendering Optimizer

Usage:
  uv run train.py --scene scenes/cornell_box.json
  uv run train.py --load-targets targets/cornell_box --scene scenes/cornellbox.json
  uv run train.py --scene scenes/cornellbox.json --optimize-geometry --no-gui
  uv run train.py --list-scenes
"""

import glob
import json
import os
import sys
import time

import numpy as np

# --- CLI ---
use_gui = True
optimize_geometry = False
scene_path = None
load_targets_dir = None

args = sys.argv[1:]
for i, arg in enumerate(args):
    if arg == "--no-gui":
        use_gui = False
    elif arg == "--optimize-geometry":
        optimize_geometry = True
    elif arg == "--scene" and i + 1 < len(args):
        scene_path = args[i + 1]
    elif arg == "--load-targets" and i + 1 < len(args):
        load_targets_dir = args[i + 1]
    elif arg == "--list-scenes":
        for f in sorted(glob.glob("scenes/*.json")):
            with open(f) as fh:
                d = json.load(fh)
            print(f"  {f:40s} — {d.get('name', '?')}")
        sys.exit(0)

if scene_path is None:
    for c in ["scenes/cornellbox.json", "cornellbox.json"]:
        if os.path.exists(c):
            scene_path = c
            break
    if scene_path is None:
        print("Error: --scene required")
        sys.exit(1)

# --- Taichi init (must be before package import) ---
import taichi as ti

ti.init(arch=ti.gpu)

# --- Imports (fields allocated on import of diffpt.fields) ---

from diffpt.config import (
    FD_EPS,
    GEO_LR,
    IMG_H,
    IMG_W,
    LR,
    MAX_SPHERES,
    MOMENTUM,
    NUM_ITERS,
    SPP,
    SPP_FD,
)
from diffpt.fields import (
    DISP_H,
    DISP_W,
    adam_t,
    albedo,
    albedo_grad,
    current_view,
    display_image,
    emission,
    emission_grad,
    fields_dict,
    image,
    image_fd,
    loss,
    loss_fd,
    metallic,
    metallic_grad,
    num_spheres_field,
    num_views_field,
    roughness,
    roughness_grad,
    sphere_center,
    sphere_radius,
    target_images,
    zero_grads,
    zero_velocities,
)
from diffpt.kernels import (
    adam_step_albedo,
    adam_step_emission,
    adam_step_roughness_metallic,
    apply_geo_adam_step,
    compose_display,
    compute_loss_fd_view,
    compute_loss_view,
    do_render,
    render_and_grad_kernel,
    render_fd_kernel,
)
from diffpt.scene_loader import (
    apply_ground_truth,
    apply_initial_guess,
    apply_initial_guess_geometry,
    load_scene,
)

# --- Hyperparameters (mutable) ---
lr_val = LR
mom_val = MOMENTUM
geo_lr_val = GEO_LR
num_iters = NUM_ITERS
spp = SPP

# --- Geometry optimizer (conditional) ---
geo_opt = None


# Material FD (SPSA for roughness/metallic)
def _render_all_views_loss_fd(info):
    # Challenge 4
    total = 0.0

    return total


def compute_roughness_metallic_grads(info):
    # Challenge 5
    """SPSA for roughness and metallic with separate perturbation sizes."""

    ...


albedo_spsa_grad = None


def compute_albedo_spsa_single_material(info, mat_idx):
    # Challenge 6
    """SPSA for one material's albedo only. Cleaner gradient than
    perturbing all materials simultaneously."""

    ...


# Optimization step — 4-phase training
#   Phase 0 (emission) — emission only
#   Phase 1 (albedo) — albedo + emission
#   Phase 2 (full): — albedo + roughness/metallic
#   Phase 3 (finetune): — all params, reduced LR

scheduler = None
_prev_loss = float("inf")
_best_params = None
_roughness_momentum_reset = False


def _snapshot_params(info):
    return {
        "albedo": [albedo[i].to_numpy().tolist() for i in range(info.num_materials)],
        "roughness": [roughness[i] for i in range(info.num_materials)],
        "metallic": [metallic[i] for i in range(info.num_materials)],
        "emission": emission[0].to_numpy().tolist(),
    }


def _restore_params(info, snap):
    for i in range(info.num_materials):
        albedo[i] = snap["albedo"][i]
        roughness[i] = snap["roughness"][i]
        metallic[i] = snap["metallic"][i]
    emission[0] = snap["emission"]


def optimization_step(info, iteration):

    # Geometry (skip during emission phase)
    if optimize_geometry and geo_opt is not None and phase != "emission":

        def spsa_render():
            image_fd.fill(0.0)
            render_fd_kernel()

        def spsa_loss(v):
            loss_fd[None] = 0.0
            compute_loss_fd_view(v)
            return loss_fd[None]

        geo_opt.compute_spsa_gradients(info, spsa_render, spsa_loss, current_view)
        geo_opt.step(geo_lr_val, iteration)
        from diffpt.geo_optimizer import (
            sphere_center_grad_combined,
            sphere_radius_grad_combined,
        )

        apply_geo_adam_step(sphere_center_grad_combined, sphere_radius_grad_combined)

    current_loss = loss[None]
    scheduler.track_loss(iteration, current_loss)

    if _best_params is not None and current_loss > _prev_loss * 1.5 and iteration > 30:
        _restore_params(info, _best_params)
        from diffpt.fields import (
            albedo_adam_m,
            albedo_adam_v,
            emission_adam_m,
            emission_adam_v,
            metallic_adam_m,
            metallic_adam_v,
            roughness_adam_m,
            roughness_adam_v,
        )

        albedo_adam_m.fill(0.0)
        albedo_adam_v.fill(0.0)
        emission_adam_m.fill(0.0)
        emission_adam_v.fill(0.0)
        roughness_adam_m.fill(0.0)
        roughness_adam_v.fill(0.0)
        metallic_adam_m.fill(0.0)
        metallic_adam_v.fill(0.0)
        current_loss = _prev_loss

        if _best_params is None or current_loss < _prev_loss:
            _best_params = _snapshot_params(info)
            _prev_loss = current_loss
            return current_loss


# Target loading


def _populate_target_field(view_idx, arr):
    """Copy a (W, H, 3) numpy array into target_images[view_idx]."""
    for y in range(min(arr.shape[1], IMG_H)):
        for x in range(min(arr.shape[0], IMG_W)):
            target_images[view_idx, x, y] = arr[x, y].tolist()


# Ignore, this function just loads in a rendered targets image data
# as a numpy array and fills in the fields struct for all relevant scene data
def load_or_render_targets(info):
    if load_targets_dir:
        print(f"\nLoading targets from {load_targets_dir}/...")

        # Check if directory contains view_0.npy (synthetic) or real images
        has_npy = os.path.exists(os.path.join(load_targets_dir, "view_0.npy"))

        if has_npy:
            # Synthetic targets from render_targets.py
            for v in range(info.num_views):
                npy = os.path.join(load_targets_dir, f"view_{v}.npy")
                if os.path.exists(npy):
                    arr = np.load(npy)
                    _populate_target_field(v, arr)
                    print(f"  View {v}: loaded {npy}")
                else:
                    print(f"  View {v}: {npy} missing — rendering fallback")
                    apply_ground_truth(info, fields_dict)
                    do_render(spp * 2, v)
                    arr = image.to_numpy()
                    _populate_target_field(v, arr)

    else:
        # Render synthetic targets
        print("\nRendering targets...")
        apply_ground_truth(info, fields_dict)
        for v in range(info.num_views):
            do_render(spp * 2, v)
            arr = image.to_numpy()
            _populate_target_field(v, arr)
            save_ppm(f"output/target_v{v}.ppm", image)
    print(f"  {info.num_views} views ready.\n")


def save_ppm(filename, img_field):
    arr = img_field.to_numpy()
    arr = np.clip(arr, 0.0, 1.0)
    arr = np.power(arr, 1.0 / 2.2)
    arr = (arr * 255).astype(np.uint8)
    h, w, _ = arr.shape
    with open(filename, "wb") as f:
        f.write(f"P6\n{w} {h}\n255\n".encode())
        f.write(arr.tobytes())


# GUI loop
def run_gui(info):
    global lr_val, mom_val, geo_lr_val
    display_view = 0
    window = ti.ui.Window(
        f"Diff PT — {info.name}", res=(DISP_W * 2, DISP_H * 2), vsync=True
    )
    canvas = window.get_canvas()
    gui = window.get_gui()
    iteration = 0
    paused = False
    loss_history = []
    t0 = time.time()
    print("GGUI: SPACE=pause R=reset V=view ESC=quit")

    while window.running:
        if window.get_event(ti.ui.PRESS):
            if window.event.key == ti.ui.ESCAPE:
                break
            elif window.event.key == " ":
                paused = not paused
            elif window.event.key == "v":
                display_view = (display_view + 1) % info.num_views
            elif window.event.key == "r":
                apply_initial_guess(info, fields_dict)
                if optimize_geometry:
                    apply_initial_guess_geometry(info, fields_dict)
                    if geo_opt:
                        geo_opt.reset()
                if scheduler:
                    scheduler.reset()
            global _prev_loss, _best_params, _roughness_momentum_reset
            _prev_loss = float("inf")
            _best_params = None
            _roughness_momentum_reset = False
            zero_velocities()
            iteration = 0
            loss_history.clear()
            t0 = time.time()

        lrv = -1
        if not paused and iteration < num_iters:
            cl = optimization_step(info, iteration)
            loss_history.append(cl)
            lrv = info.num_views - 1
            if iteration % 10 == 0:
                print(
                    f"  Iter {iteration:4d} | Loss: {cl:.6f} | {time.time() - t0:.1f}s"
                )
            if iteration % 50 == 0:
                for v in range(info.num_views):
                    do_render(spp, v)
                    save_ppm(f"output/iter_{iteration:04d}_v{v}.ppm", image)
                lrv = info.num_views - 1
            iteration += 1
        elif not paused and iteration == num_iters:
            for v in range(info.num_views):
                do_render(spp, v)
                save_ppm(f"output/final_v{v}.ppm", image)
            if loss_history:
                np.save("output/loss_history.npy", np.array(loss_history))
            print(f"\n  Done! Loss: {loss_history[-1]:.6f}")
            lrv = info.num_views - 1
            iteration += 1

        if display_view != lrv or paused or iteration > num_iters:
            ds = spp if (paused or iteration > num_iters) else max(SPP_FD, 2)
            do_render(ds, display_view)
        compose_display(display_view)
        canvas.set_image(display_image)

        with gui.sub_window("Controls", x=0.01, y=0.01, width=0.35, height=0.92):
            gui.text(f"Scene: {info.name}")
            gui.text(f"Iter: {iteration}/{num_iters}")
            if loss_history:
                gui.text(f"Loss: {loss_history[-1]:.6f}")
                if len(loss_history) > 1:
                    gui.text(
                        f"Reduction: {(1 - loss_history[-1] / loss_history[0]) * 100:.1f}%"
                    )
            gui.text(f"View: {display_view}/{info.num_views} (V)")
            gui.text("Left: TARGET | Right: CURRENT")
            if scheduler:
                gui.text(scheduler.status_string(iteration))
            gui.text("")
            lr_val = gui.slider_float("Learning Rate", lr_val, 0.001, 0.1)
            if optimize_geometry:
                geo_lr_val = gui.slider_float("Geo LR", geo_lr_val, 0.0005, 0.05)
            gui.text("")
            gui.text("--- Materials ---")
            for i in range(info.num_materials):
                a = albedo[i]
                r = roughness[i]
                m = metallic[i]
                gui.text(
                    f" {info.material_names[i]}: ({a[0]:.2f},{a[1]:.2f},{a[2]:.2f}) r={r:.2f} m={m:.2f}"
                )
            e = emission[0]
            gui.text(f" emit: ({e[0]:.1f},{e[1]:.1f},{e[2]:.1f})")
            if optimize_geometry:
                gui.text("")
                gui.text("--- Geometry ---")
                for i in range(info.num_spheres):
                    c = sphere_center[i]
                    r = sphere_radius[i]
                    gui.text(f" s{i}: ({c[0]:.2f},{c[1]:.2f},{c[2]:.2f}) r={r:.2f}")
        window.show()
    window.destroy()


def run_headless(info):
    loss_history = []
    t0 = time.time()
    for it in range(num_iters):
        cl = optimization_step(info, it)
        loss_history.append(cl)
        if it % 10 == 0 or it == num_iters - 1:
            print(f"  Iter {it:4d} | Loss: {cl:.6f} | {time.time() - t0:.1f}s")
        if it % 50 == 0 or it == num_iters - 1:
            for v in range(info.num_views):
                do_render(spp, v)
                save_ppm(f"output/iter_{it:04d}_v{v}.ppm", image)
    np.save("output/loss_history.npy", np.array(loss_history))
    print(f"\n  Final loss: {loss_history[-1]:.6f}")


# Main

os.makedirs("output", exist_ok=True)
info = load_scene(scene_path, fields_dict)
s = info.settings
num_iters = s.get("num_iters", num_iters)
spp = s.get("spp", spp)

mode = f"NEE+GGX {info.num_views}-view"
if optimize_geometry:
    mode += " +geometry"
print("=" * 60)
print(f"  {info.name} — {mode}")
print(f"  {info.num_quads}q {info.num_spheres}s {info.num_materials}m")
print("=" * 60)

load_or_render_targets(info)
apply_initial_guess(info, fields_dict)
if optimize_geometry:
    apply_initial_guess_geometry(info, fields_dict)
    from diffpt.geo_optimizer import GeoOptimizer

    geo_opt = GeoOptimizer(info.num_spheres, fields_dict, IMG_W, IMG_H)
zero_velocities()

# Initialize training scheduler
from diffpt.scheduler import TrainingScheduler

scheduler = TrainingScheduler(
    spp, info.num_views, num_iters, lr_val, info.num_materials
)
print(
    f"  Training phases: emission(0-{scheduler.emission_phase_end - 1}), "
    f"albedo({scheduler.emission_phase_end}-{scheduler.albedo_phase_end - 1}), "
    f"full({scheduler.albedo_phase_end}-{scheduler.brdf_phase_end - 1}), "
    f"finetune({scheduler.brdf_phase_end}+)"
)

if use_gui:
    run_gui(info)
else:
    run_headless(info)
