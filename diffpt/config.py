"""
diffpt/config.py — Constants and configuration defaults.
"""

# Compile-time maximums (Taichi field shape bounds)
MAX_QUADS = 16
MAX_SPHERES = 8
MAX_MATERIALS = 16
MAX_VIEWS = 8
MAX_DEPTH = 6

# Math
EPS = 1e-6
PI = 3.14159265358979

# Finite difference
FD_EPS = 0.005

# Defaults (overridden by scene JSON or CLI)
IMG_W = 256
IMG_H = 256
SPP = 8
SPP_FD = 2
NUM_ITERS = 500
LR = 0.015
MOMENTUM = 0.9
GEO_LR = 0.005
