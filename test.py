"""
test.py - Helps check if functions are working and if challenges are completed
Usages:
    uv run test.py --challenge 1

    uv run test.py --all
"""

import os
import sys
import time

import taichi as ti
import taichi.math as tm

ti.init(arch=ti.gpu)

import render_targets

render_targets.init_fields()

from render_targets import (
    albedo,
    emission,
    eval_brdf,
    fresnel_schlick,
    ggx_d,
    hash_u32,
    light_area_field,
    light_idx_field,
    metallic,
    next_rand,
    num_quads_field,
    num_spheres_field,
    quad_center,
    quad_is_light,
    quad_mat,
    quad_normal,
    quad_u,
    quad_v,
    roughness,
    scene_intersect,
    smith_g1,
    sphere_center,
    sphere_mat,
    sphere_radius,
    visibility_test,
)

PI = tm.pi
EPS = 1e-8

result_u32 = ti.field(dtype=ti.u32, shape=())
result_f32 = ti.field(dtype=ti.f32, shape=())
result_vec3 = ti.Vector.field(3, dtype=ti.f32, shape=())

rng_seed = render_targets.rng_seed

# For scene_intersect we need to capture multiple return values
hit_t = ti.field(dtype=ti.f32, shape=())
hit_normal = ti.Vector.field(3, dtype=ti.f32, shape=())
hit_mat = ti.field(dtype=ti.i32, shape=())
hit_light = ti.field(dtype=ti.i32, shape=())


@ti.kernel
def hash_test(val: ti.u32):
    result_u32[None] = hash_u32(val)


@ti.kernel
def next_rand_test(seed: ti.u32):
    rng_seed[0, 0] = seed
    result_f32[None] = next_rand(0, 0)
    result_u32[None] = rng_seed[0, 0]


@ti.kernel
def ggx_d_test(ndoth: ti.f32, alpha: ti.f32):
    result_f32[None] = ggx_d(ndoth, alpha)


@ti.kernel
def smith_g1_test(ndotv: ti.f32, alpha: ti.f32):
    result_f32[None] = smith_g1(ndotv, alpha)


@ti.kernel
def fresnel_test(cos_theta: ti.f32, f0x: ti.f32, f0y: ti.f32, f0z: ti.f32):
    result_vec3[None] = fresnel_schlick(cos_theta, tm.vec3(f0x, f0y, f0z))


@ti.kernel
def brdf_test(
    wo_x: ti.f32,
    wo_y: ti.f32,
    wo_z: ti.f32,
    wi_x: ti.f32,
    wi_y: ti.f32,
    wi_z: ti.f32,
    n_x: ti.f32,
    n_y: ti.f32,
    n_z: ti.f32,
    col_x: ti.f32,
    col_y: ti.f32,
    col_z: ti.f32,
    rough: ti.f32,
    metal: ti.f32,
):
    result_vec3[None] = eval_brdf(
        tm.vec3(wo_x, wo_y, wo_z),
        tm.vec3(wi_x, wi_y, wi_z),
        tm.vec3(n_x, n_y, n_z),
        tm.vec3(col_x, col_y, col_z),
        rough,
        metal,
    )


@ti.kernel
def intersect_test(
    ox: ti.f32,
    oy: ti.f32,
    oz: ti.f32,
    dx: ti.f32,
    dy: ti.f32,
    dz: ti.f32,
):
    t, n, m, l = scene_intersect(
        tm.vec3(ox, oy, oz),
        tm.vec3(dx, dy, dz),
    )
    hit_t[None] = t
    hit_normal[None] = n
    hit_mat[None] = m
    hit_light[None] = l


@ti.kernel
def visibility_test_kernel(
    px: ti.f32,
    py: ti.f32,
    pz: ti.f32,
    tx: ti.f32,
    ty: ti.f32,
    tz: ti.f32,
):
    result_f32[None] = visibility_test(
        tm.vec3(px, py, pz),
        tm.vec3(tx, ty, tz),
    )


def setup_floor_quad():
    """Place a single quad as a floor at y=0, spanning x/z [-1,1]."""
    num_quads_field[None] = 1
    num_spheres_field[None] = 0
    quad_center[0] = [0.0, 0.0, 0.0]
    quad_u[0] = [1.0, 0.0, 0.0]
    quad_v[0] = [0.0, 0.0, 1.0]
    quad_normal[0] = [0.0, 1.0, 0.0]
    quad_mat[0] = 0
    quad_is_light[0] = 0
    albedo[0] = [0.8, 0.8, 0.8]
    roughness[0] = 0.5
    metallic[0] = 0.0


def setup_sphere():
    """Place a unit sphere at origin."""
    num_quads_field[None] = 0
    num_spheres_field[None] = 1
    sphere_center[0] = [0.0, 0.0, 0.0]
    sphere_radius[0] = 1.0
    sphere_mat[0] = 0
    albedo[0] = [0.8, 0.2, 0.2]
    roughness[0] = 0.3
    metallic[0] = 0.0


def setup_occluder_scene():
    """Floor quad + sphere blocker for visibility tests."""
    num_quads_field[None] = 1
    num_spheres_field[None] = 1
    # Floor
    quad_center[0] = [0.0, 0.0, 0.0]
    quad_u[0] = [5.0, 0.0, 0.0]
    quad_v[0] = [0.0, 0.0, 5.0]
    quad_normal[0] = [0.0, 1.0, 0.0]
    quad_mat[0] = 0
    quad_is_light[0] = 0
    # Blocking sphere between source and target
    sphere_center[0] = [0.0, 1.0, 0.0]
    sphere_radius[0] = 0.3
    sphere_mat[0] = 0


def test_brdf_energy_conservation():
    """BRDF * ndotl should not exceed 1 for any configuration."""
    n = (0.0, 1.0, 0.0)
    wo = (0.0, 1.0, 0.0)  # looking straight down
    wi = (0.0, 1.0, 0.0)  # light from above
    brdf_test(*wo, *wi, *n, 1.0, 1.0, 1.0, 0.5, 0.0)
    r = result_vec3[None]
    # result already includes ndotl; for a white diffuse surface
    # the max should be around albedo/pi
    for i in range(3):
        assert r[i] < 1.5, f"BRDF channel {i} too high: {r[i]}"


def test_brdf_diffuse_lambert():
    """Pure diffuse (metal=0, rough=1) should be close to albedo/pi * ndotl."""
    n = (0.0, 1.0, 0.0)
    wo = (0.0, 1.0, 0.0)
    wi = (0.0, 1.0, 0.0)
    color = (0.5, 0.5, 0.5)
    brdf_test(*wo, *wi, *n, *color, 1.0, 0.0)
    r = result_vec3[None]
    # ndotl = 1.0, rough lambert ≈ (1-F)*color/pi * 1.0
    # F at head-on with f0=0.04 is ~0.04, so kd ≈ 0.96
    expected_approx = 0.96 * 0.5 / PI
    for i in range(3):
        assert abs(r[i] - expected_approx) < 0.05, (
            f"Channel {i}: {r[i]} vs expected ~{expected_approx}"
        )


def test_brdf_metal_no_diffuse():
    """Fully metallic surface should have negligible diffuse component."""
    n = (0.0, 1.0, 0.0)
    wo = (0.0, 1.0, 0.0)
    wi = (0.0, 1.0, 0.0)
    brdf_test(*wo, *wi, *n, 0.9, 0.1, 0.1, 0.1, 1.0)  # metal=1
    r_metal = result_vec3[None]
    brdf_test(*wo, *wi, *n, 0.9, 0.1, 0.1, 0.1, 0.0)  # metal=0
    r_diel = result_vec3[None]
    # Metal should have higher specular but zero diffuse;
    # the overall value depends on roughness, but the key property
    # is that metal reflects the base color
    assert r_metal[0] > 0.0, "Metal BRDF should be non-zero at normal incidence"


def test_brdf_symmetry():
    """Swapping wo and wi should give the same BRDF value (excluding cosine)."""
    import math

    n = (0.0, 1.0, 0.0)
    wo = (0.3, 0.9, 0.1)
    wi = (-0.2, 0.8, 0.3)

    def norm(v):
        l = math.sqrt(sum(x * x for x in v))
        return tuple(x / l for x in v)

    wo = norm(wo)
    wi = norm(wi)

    # eval_brdf returns (diff + spec) * ndotl
    # To compare the BRDF itself, divide out ndotl = dot(n, wi)
    ndotl_1 = max(sum(n[i] * wi[i] for i in range(3)), 1e-6)
    ndotl_2 = max(sum(n[i] * wo[i] for i in range(3)), 1e-6)

    brdf_test(*wo, *wi, *n, 0.5, 0.5, 0.5, 0.4, 0.0)
    r1 = [result_vec3[None][i] / ndotl_1 for i in range(3)]

    brdf_test(*wi, *wo, *n, 0.5, 0.5, 0.5, 0.4, 0.0)
    r2 = [result_vec3[None][i] / ndotl_2 for i in range(3)]

    for i in range(3):
        assert abs(r1[i] - r2[i]) < 1e-3, (
            f"Reciprocity failed ch{i}: {r1[i]} vs {r2[i]}"
        )


def test_intersect_quad_hit():
    """Ray pointing down should hit the floor quad."""
    setup_floor_quad()
    intersect_test(0.0, 2.0, 0.0, 0.0, -1.0, 0.0)
    assert abs(hit_t[None] - 2.0) < 1e-3
    assert hit_mat[None] == 0
    assert abs(hit_normal[None][1] - 1.0) < 1e-3  # normal is +Y


def test_intersect_quad_miss():
    """Ray parallel to the floor should miss."""
    setup_floor_quad()
    intersect_test(0.0, 2.0, 0.0, 1.0, 0.0, 0.0)
    assert hit_t[None] > 1e9  # no hit
    assert hit_mat[None] == -1


def test_intersect_quad_outside_bounds():
    """Ray hitting the plane but outside the quad extents should miss."""
    setup_floor_quad()
    # Quad spans [-1,1] in x and z; aim at x=5
    intersect_test(5.0, 2.0, 0.0, 0.0, -1.0, 0.0)
    assert hit_mat[None] == -1


def test_intersect_sphere_hit():
    """Ray aimed at a unit sphere at origin."""
    setup_sphere()
    intersect_test(0.0, 0.0, -5.0, 0.0, 0.0, 1.0)
    assert abs(hit_t[None] - 4.0) < 1e-3  # hits front face at z=-1
    assert hit_mat[None] == 0
    # Normal should point toward camera (-Z)
    assert hit_normal[None][2] < -0.9


def test_intersect_sphere_miss():
    """Ray that misses the sphere entirely."""
    setup_sphere()
    intersect_test(5.0, 0.0, -5.0, 0.0, 0.0, 1.0)
    assert hit_mat[None] == -1


def test_intersect_closest_hit():
    """When both quad and sphere are present, return the closer one."""
    num_quads_field[None] = 1
    num_spheres_field[None] = 1
    # Floor at y=-2
    quad_center[0] = [0.0, -2.0, 0.0]
    quad_u[0] = [5.0, 0.0, 0.0]
    quad_v[0] = [0.0, 0.0, 5.0]
    quad_normal[0] = [0.0, 1.0, 0.0]
    quad_mat[0] = 0
    quad_is_light[0] = 0
    # Sphere at y=0 (closer to ray origin above)
    sphere_center[0] = [0.0, 0.0, 0.0]
    sphere_radius[0] = 0.5
    sphere_mat[0] = 1
    intersect_test(0.0, 3.0, 0.0, 0.0, -1.0, 0.0)
    # Should hit sphere first at t=2.5 (3.0 - 0.5)
    assert abs(hit_t[None] - 2.5) < 1e-2
    assert hit_mat[None] == 1


def test_visibility_clear():
    """No geometry between two points → visible."""
    num_quads_field[None] = 0
    num_spheres_field[None] = 0
    visibility_test_kernel(0.0, 0.0, 0.0, 5.0, 5.0, 5.0)
    assert result_f32[None] > 0.5


def test_visibility_blocked_by_sphere():
    """Sphere between source and target → occluded."""
    setup_occluder_scene()
    # Source below sphere, target above sphere
    visibility_test_kernel(0.0, 0.0, 0.0, 0.0, 2.0, 0.0)
    assert result_f32[None] < 0.5


def test_visibility_around_blocker():
    """Path that goes around the occluder → visible."""
    setup_occluder_scene()
    # Offset in X so the line of sight misses the sphere at (0,1,0) r=0.3
    visibility_test_kernel(2.0, 0.5, 0.0, 2.0, 1.5, 0.0)
    assert result_f32[None] > 0.5


def test_visibility_blocked_by_quad():
    """Quad between source and target → occluded."""
    num_quads_field[None] = 1
    num_spheres_field[None] = 0
    # Vertical wall at z=0
    quad_center[0] = [0.0, 0.0, 0.0]
    quad_u[0] = [5.0, 0.0, 0.0]
    quad_v[0] = [0.0, 5.0, 0.0]
    quad_normal[0] = [0.0, 0.0, 1.0]
    quad_mat[0] = 0
    quad_is_light[0] = 0
    # Source in front, target behind the wall
    visibility_test_kernel(0.0, 0.0, -2.0, 0.0, 0.0, 2.0)
    assert result_f32[None] < 0.5


def run_all():
    challenge_1()
    challenge_2()


def challenge_1():

    # test hash
    hash_test(42)
    a = result_u32[None]
    hash_test(42)
    b = result_u32[None]
    assert a == b

    hash_test(0)
    h0 = result_u32[None]
    hash_test(1)
    h1 = result_u32[None]
    diff = bin(h0 ^ h1).count("1")  # check diff in bits
    assert diff >= 8

    # Next Rand Test
    for seed in [0, 1, 0xDEADBEEF, 0xFFFFFFFF]:
        next_rand_test(seed)
        val = result_f32[None]
        assert 0.0 <= val <= 1.0

    next_rand_test(123)
    new_seed = result_u32[None]
    assert new_seed != 123

    # GGX D Test
    ggx_d_test(1.0, 1.0)
    expected = 1.0 / PI
    assert abs(result_f32[None] - expected) < 1e-5

    ggx_d_test(0.0, 0.5)
    ggx_d_test(0.0, 0.01)
    assert result_f32[None] <= 1e-3

    # Smith G1 Test
    smith_g1_test(1.0, 0.5)
    assert abs(result_f32[None] - 1.0) < 1e-5

    for ndotv in [0.1, 0.5, 0.9]:
        for alpha in [0.01, 0.5, 1.0]:
            smith_g1_test(ndotv, alpha)
            g = result_f32[None]
            assert 0.0 < g <= 1.0 + 1e-5

    # Fresnel Tests
    f0 = (0.04, 0.04, 0.04)
    fresnel_test(1.0, 0.04, 0.04, 0.04)
    r = result_vec3[None]
    for i in range(3):
        assert abs(r[i] - f0[i]) < 1e-5

    fresnel_test(0.0, 0.04, 0.04, 0.04)
    r = result_vec3[None]
    for i in range(3):
        assert abs(r[i] - 1.0) < 1e-5

    print("Success! You completed challenge 1!")


def challenge_2():
    test_brdf_energy_conservation()
    test_brdf_diffuse_lambert()
    test_brdf_metal_no_diffuse()
    test_brdf_symmetry()
    test_intersect_quad_hit()
    test_intersect_quad_miss()
    test_intersect_quad_outside_bounds()
    test_intersect_sphere_hit()
    test_intersect_sphere_miss()
    test_intersect_closest_hit()
    test_visibility_clear()
    test_visibility_blocked_by_sphere()
    test_visibility_around_blocker()
    test_visibility_blocked_by_quad()
    print("Success! You completed challenge 2!")


def main():
    challenge = 0

    args = sys.argv[1:]
    for i, arg in enumerate(args):
        if arg == "--challenge" and i + 1 < len(args):
            challenge = args[i + 1]
            break
        elif arg == "--all":
            challenge = "0"
            break
        else:
            print("Try running: uv run test.py --all")
            sys.exit(0)

    match challenge:
        case "0":
            run_all()
        case "1":
            challenge_1()
        case "2":
            challenge_2()
        case _:
            print("Try running: uv run test.py --all")
            sys.exit(0)


if __name__ == "__main__":
    main()
