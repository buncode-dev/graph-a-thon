"""
diffpt/fields.py — Central Taichi field allocation.

All Taichi fields are allocated here and imported by other modules.
This ensures a single source of truth and avoids circular dependencies.

Taichi constraint: @ti.func and @ti.kernel can reference fields from any
module as long as they are allocated before the first kernel call.
"""

import taichi as ti
import taichi.math as tm

from .config import *

# Runtime counts (set by scene_loader from JSON)

num_quads_field = ti.field(dtype=ti.i32, shape=())
num_spheres_field = ti.field(dtype=ti.i32, shape=())
num_materials_field = ti.field(dtype=ti.i32, shape=())
num_views_field = ti.field(dtype=ti.i32, shape=())
light_idx_field = ti.field(dtype=ti.i32, shape=())
light_area_field = ti.field(dtype=ti.f32, shape=())


# Materials

albedo = ti.Vector.field(3, dtype=ti.f32, shape=(MAX_MATERIALS,))
albedo_grad = ti.Vector.field(3, dtype=ti.f32, shape=(MAX_MATERIALS,))
roughness = ti.field(dtype=ti.f32, shape=(MAX_MATERIALS,))
roughness_grad = ti.field(dtype=ti.f32, shape=(MAX_MATERIALS,))
metallic = ti.field(dtype=ti.f32, shape=(MAX_MATERIALS,))
metallic_grad = ti.field(dtype=ti.f32, shape=(MAX_MATERIALS,))
emission = ti.Vector.field(3, dtype=ti.f32, shape=(1,))
emission_grad = ti.Vector.field(3, dtype=ti.f32, shape=(1,))


# Geometry — quads

quad_center = ti.Vector.field(3, dtype=ti.f32, shape=(MAX_QUADS,))
quad_u = ti.Vector.field(3, dtype=ti.f32, shape=(MAX_QUADS,))
quad_v = ti.Vector.field(3, dtype=ti.f32, shape=(MAX_QUADS,))
quad_normal = ti.Vector.field(3, dtype=ti.f32, shape=(MAX_QUADS,))
quad_mat = ti.field(dtype=ti.i32, shape=(MAX_QUADS,))
quad_is_light = ti.field(dtype=ti.i32, shape=(MAX_QUADS,))


# Geometry — spheres

sphere_center = ti.Vector.field(3, dtype=ti.f32, shape=(MAX_SPHERES,))
sphere_radius = ti.field(dtype=ti.f32, shape=(MAX_SPHERES,))
sphere_mat = ti.field(dtype=ti.i32, shape=(MAX_SPHERES,))
sphere_center_grad = ti.Vector.field(3, dtype=ti.f32, shape=(MAX_SPHERES,))
sphere_radius_grad = ti.field(dtype=ti.f32, shape=(MAX_SPHERES,))
sphere_center_vel = ti.Vector.field(3, dtype=ti.f32, shape=(MAX_SPHERES,))
sphere_radius_vel = ti.field(dtype=ti.f32, shape=(MAX_SPHERES,))


# Cameras

cam_pos = ti.Vector.field(3, dtype=ti.f32, shape=(MAX_VIEWS,))
cam_fwd = ti.Vector.field(3, dtype=ti.f32, shape=(MAX_VIEWS,))
cam_right = ti.Vector.field(3, dtype=ti.f32, shape=(MAX_VIEWS,))
cam_up_vec = ti.Vector.field(3, dtype=ti.f32, shape=(MAX_VIEWS,))
current_view = ti.field(dtype=ti.i32, shape=())


# Image buffers

image = ti.Vector.field(3, dtype=ti.f32, shape=(IMG_W, IMG_H))
image_fd = ti.Vector.field(3, dtype=ti.f32, shape=(IMG_W, IMG_H))
target_images = ti.Vector.field(3, dtype=ti.f32, shape=(MAX_VIEWS, IMG_W, IMG_H))
loss = ti.field(dtype=ti.f32, shape=())
loss_fd = ti.field(dtype=ti.f32, shape=())
rng_seed = ti.field(dtype=ti.u32, shape=(IMG_W, IMG_H))

# Display (side-by-side target | current)
DISP_W = IMG_W * 2
DISP_H = IMG_H
display_image = ti.Vector.field(3, dtype=ti.f32, shape=(DISP_W, DISP_H))


# Adam optimizer state (first moment m, second moment v)

# Albedo (vec3 per material)
albedo_adam_m = ti.Vector.field(3, dtype=ti.f32, shape=(MAX_MATERIALS,))
albedo_adam_v = ti.Vector.field(3, dtype=ti.f32, shape=(MAX_MATERIALS,))

# Emission (vec3, single)
emission_adam_m = ti.Vector.field(3, dtype=ti.f32, shape=(1,))
emission_adam_v = ti.Vector.field(3, dtype=ti.f32, shape=(1,))

# Roughness (scalar per material)
roughness_adam_m = ti.field(dtype=ti.f32, shape=(MAX_MATERIALS,))
roughness_adam_v = ti.field(dtype=ti.f32, shape=(MAX_MATERIALS,))

# Metallic (scalar per material)
metallic_adam_m = ti.field(dtype=ti.f32, shape=(MAX_MATERIALS,))
metallic_adam_v = ti.field(dtype=ti.f32, shape=(MAX_MATERIALS,))

# Adam iteration counter (shared across all params)
adam_t = ti.field(dtype=ti.i32, shape=())


# Fields dictionary (passed to scene_loader and other modules)

fields_dict = {
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


def zero_velocities():
    """Reset all Adam optimizer state."""
    albedo_adam_m.fill(0.0)
    albedo_adam_v.fill(0.0)
    emission_adam_m.fill(0.0)
    emission_adam_v.fill(0.0)
    roughness_adam_m.fill(0.0)
    roughness_adam_v.fill(0.0)
    metallic_adam_m.fill(0.0)
    metallic_adam_v.fill(0.0)
    sphere_center_vel.fill(0.0)
    sphere_radius_vel.fill(0.0)
    adam_t[None] = 0


def zero_grads():
    """Reset all gradient accumulators."""
    albedo_grad.fill(0.0)
    emission_grad.fill(0.0)
    roughness_grad.fill(0.0)
    metallic_grad.fill(0.0)
    sphere_center_grad.fill(0.0)
    sphere_radius_grad.fill(0.0)
    loss[None] = 0.0
