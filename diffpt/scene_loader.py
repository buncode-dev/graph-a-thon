"""
scene_loader.py — Load scene descriptions from JSON into Taichi fields.

The loader resolves material names → indices, validates geometry, and
populates all the Taichi fields that the renderer reads from.

Usage:
    from scene_loader import load_scene, SceneInfo
    info = load_scene("scenes/cornell_box.json", taichi_fields_dict)
"""

import json
import os

import numpy as np


class SceneInfo:
    """Holds metadata extracted from a scene file, used by the renderer."""

    def __init__(self):
        self.name = ""
        self.description = ""
        self.num_quads = 0
        self.num_spheres = 0
        self.num_materials = 0
        self.num_views = 0
        self.light_idx = -1
        self.light_area = 0.0
        self.material_names = []
        self.settings = {}
        self.gt_materials = {}  # ground truth material params
        self.gt_emission = [18.0, 18.0, 18.0]
        self.gt_sphere_centers = []
        self.gt_sphere_radii = []
        self.initial_guess = {}
        self.initial_guess_geometry = {}
        self.cam_positions = []
        self.cam_targets = []


def load_scene(path: str, fields: dict) -> SceneInfo:
    """Load a scene JSON file and populate Taichi fields.

    Args:
        path: Path to scene JSON file.
        fields: Dictionary of Taichi fields to populate, keyed by name:
            quad_center, quad_u, quad_v, quad_normal, quad_mat, quad_is_light,
            sphere_center, sphere_radius, sphere_mat,
            albedo, roughness, metallic, emission,
            cam_pos, cam_fwd, cam_right, cam_up_vec,
            num_quads_field, num_spheres_field, num_materials_field,
            num_views_field, light_idx_field, light_area_field

    Returns:
        SceneInfo with metadata.
    """
    with open(path, "r") as f:
        data = json.load(f)

    info = SceneInfo()
    info.name = data.get("name", os.path.basename(path))
    info.description = data.get("description", "")
    info.settings = data.get("settings", {})
    info.gt_emission = data.get("emission", [18.0, 18.0, 18.0])

    # --- Resolve materials ---
    mat_defs = data.get("materials", {})
    mat_name_to_idx = {}
    mat_names = []
    idx = 0
    for name, props in mat_defs.items():
        mat_name_to_idx[name] = idx
        mat_names.append(name)
        idx += 1
    info.num_materials = len(mat_names)
    info.material_names = mat_names
    info.gt_materials = mat_defs

    # Populate material fields
    for name, midx in mat_name_to_idx.items():
        props = mat_defs[name]
        fields["albedo"][midx] = props.get("albedo", [0.5, 0.5, 0.5])
        fields["roughness"][midx] = props.get("roughness", 0.5)
        fields["metallic"][midx] = props.get("metallic", 0.0)

    fields["emission"][0] = info.gt_emission

    # --- Quads ---
    quads = data.get("quads", [])
    info.num_quads = len(quads)
    info.light_idx = -1
    for i, q in enumerate(quads):
        fields["quad_center"][i] = q["center"]
        fields["quad_u"][i] = q["u"]
        fields["quad_v"][i] = q["v"]
        fields["quad_normal"][i] = q["normal"]
        mat_name = q.get("material", mat_names[0] if mat_names else "default")
        fields["quad_mat"][i] = mat_name_to_idx.get(mat_name, 0)
        is_light = q.get("is_light", False)
        fields["quad_is_light"][i] = 1 if is_light else 0
        if is_light:
            info.light_idx = i
            # Compute light area from u and v extents
            u = np.array(q["u"])
            v = np.array(q["v"])
            # Quad spans from -1*u to +1*u and -1*v to +1*v
            info.light_area = 4.0 * np.linalg.norm(u) * np.linalg.norm(v)

    # --- Spheres ---
    spheres = data.get("spheres", [])
    info.num_spheres = len(spheres)
    info.gt_sphere_centers = []
    info.gt_sphere_radii = []
    for i, s in enumerate(spheres):
        fields["sphere_center"][i] = s["center"]
        fields["sphere_radius"][i] = s["radius"]
        mat_name = s.get("material", mat_names[0] if mat_names else "default")
        fields["sphere_mat"][i] = mat_name_to_idx.get(mat_name, 0)
        info.gt_sphere_centers.append(s["center"])
        info.gt_sphere_radii.append(s["radius"])

    # --- Cameras ---
    cameras = data.get("cameras", [{"position": [0, 1, 3], "target": [0, 1, 0]}])
    info.num_views = len(cameras)
    info.cam_positions = [c["position"] for c in cameras]
    info.cam_targets = [c.get("target", [0, 1, 0]) for c in cameras]

    world_up = np.array([0.0, 1.0, 0.0])
    for v, cam in enumerate(cameras):
        pos = np.array(cam["position"], dtype=np.float32)
        tgt = np.array(cam.get("target", [0, 1, 0]), dtype=np.float32)
        fwd = tgt - pos
        fwd = fwd / (np.linalg.norm(fwd) + 1e-8)
        right = np.cross(fwd, world_up)
        right = right / (np.linalg.norm(right) + 1e-8)
        up = np.cross(right, fwd)
        fields["cam_pos"][v] = pos.tolist()
        fields["cam_fwd"][v] = fwd.tolist()
        fields["cam_right"][v] = right.tolist()
        fields["cam_up_vec"][v] = up.tolist()

    # --- Store runtime counts in Taichi scalar fields ---
    fields["num_quads_field"][None] = info.num_quads
    fields["num_spheres_field"][None] = info.num_spheres
    fields["num_materials_field"][None] = info.num_materials
    fields["num_views_field"][None] = info.num_views
    fields["light_idx_field"][None] = info.light_idx
    fields["light_area_field"][None] = info.light_area

    # --- Initial guess ---
    info.initial_guess = data.get("initial_guess", {})
    info.initial_guess_geometry = data.get("initial_guess_geometry", {})

    return info


def apply_ground_truth(info: SceneInfo, fields: dict):
    """Set ground truth material + geometry params from scene info."""
    for i, name in enumerate(info.material_names):
        props = info.gt_materials[name]
        fields["albedo"][i] = props.get("albedo", [0.5, 0.5, 0.5])
        fields["roughness"][i] = props.get("roughness", 0.5)
        fields["metallic"][i] = props.get("metallic", 0.0)
    fields["emission"][0] = info.gt_emission

    for i, c in enumerate(info.gt_sphere_centers):
        fields["sphere_center"][i] = c
        fields["sphere_radius"][i] = info.gt_sphere_radii[i]


def apply_initial_guess(info: SceneInfo, fields: dict):
    """Set initial guess from scene info with small random perturbation.

    Adding noise to the initial guess breaks the symmetry between materials
    that would otherwise start identical and receive identical gradients,
    preventing them from ever differentiating."""
    import numpy as np

    ig = info.initial_guess
    mat_ig = ig.get("materials", {})
    default_albedo = np.array(
        mat_ig.get("albedo_default", [0.5, 0.5, 0.5]), dtype=np.float32
    )
    default_rough = mat_ig.get("roughness_default", 0.5)
    default_metal = mat_ig.get("metallic_default", 0.3)

    rng = np.random.RandomState(42)  # deterministic but different per material
    for i in range(info.num_materials):
        # Perturb albedo by ±0.08 per channel
        noise = rng.uniform(-0.08, 0.08, size=3).astype(np.float32)
        perturbed_albedo = np.clip(default_albedo + noise, 0.05, 0.95)
        fields["albedo"][i] = perturbed_albedo.tolist()

        # Perturb roughness by ±0.1
        fields["roughness"][i] = np.clip(
            default_rough + rng.uniform(-0.1, 0.1), 0.05, 1.0
        )

        # Perturb metallic by ±0.1
        fields["metallic"][i] = np.clip(
            default_metal + rng.uniform(-0.1, 0.1), 0.0, 1.0
        )

    fields["emission"][0] = ig.get("emission", [10.0, 10.0, 10.0])


def apply_initial_guess_geometry(info: SceneInfo, fields: dict):
    """Set initial guess for sphere geometry."""
    ig_geo = info.initial_guess_geometry
    spheres_ig = ig_geo.get("spheres", [])
    for i, s in enumerate(spheres_ig):
        if i < info.num_spheres:
            fields["sphere_center"][i] = s.get("center", [0, 0.5, 0])
            fields["sphere_radius"][i] = s.get("radius", 0.3)
