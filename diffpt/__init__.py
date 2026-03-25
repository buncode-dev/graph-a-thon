"""
diffpt — Differentiable Path Tracer package.

Modules:
  config        — Constants and defaults
  fields        — Central Taichi field allocation
  kernels       — All Taichi @ti.func and @ti.kernel
  scene_loader  — JSON scene file parser
  geo_optimizer — Geometry gradient estimation (SPSA + Adam)
"""
