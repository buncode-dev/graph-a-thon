"""
diffpt/scheduler.py — Training schedule and performance optimizations.

Training phases:
  Phase 0 (iter 0-19):    Emission only — get overall brightness right
  Phase 1 (iter 20-79):   Albedo only — establish base colors
  Phase 2 (iter 80-349):  Albedo + roughness/metallic — full material opt
  Phase 3 (iter 350+):    Fine-tune all with reduced LR

Also manages: progressive SPP, stochastic view selection, per-material
rotating SPSA, gradient EMA, learning rate warmup + decay.
"""

import numpy as np


class TrainingScheduler:
    def __init__(self, base_spp, num_views, num_iters, base_lr, num_materials=7):
        self.base_spp = base_spp
        self.num_views = num_views
        self.num_iters = num_iters
        self.base_lr = base_lr
        self.num_materials = num_materials

        # Phase boundaries
        self.emission_phase_end = 20
        self.albedo_phase_end = 80
        self.brdf_phase_end = int(num_iters * 0.7)
        # After brdf_phase_end: fine-tune phase

        # Progressive SPP
        self.min_spp = max(2, base_spp // 4)
        self.spp_ramp_end = num_iters // 3

        # View selection
        self.min_views_per_iter = max(1, num_views // 2)
        self.full_view_interval = 10

        # Roughness/metallic SPSA interval
        self.roughness_metallic_interval = 5

        # Per-material rotating SPSA
        self._spsa_material_cursor = 0  # which material to probe next

        # LR schedule
        self.warmup_iters = 15
        self.min_lr_factor = 0.1

        # Gradient EMA
        self.grad_ema_alpha = 0.3
        self._ema_grads = {}

        # Loss tracking
        self.initial_loss = None
        self.best_loss = float("inf")

    def get_phase(self, iteration: int) -> str:
        """Return current training phase name."""
        if iteration < self.emission_phase_end:
            return "emission"
        elif iteration < self.albedo_phase_end:
            return "albedo"
        elif iteration < self.brdf_phase_end:
            return "full"
        else:
            return "finetune"

    def should_update_emission_only(self, iteration: int) -> bool:
        return iteration < self.emission_phase_end

    def should_update_albedo(self, iteration: int) -> bool:
        return iteration >= self.emission_phase_end

    def should_update_roughness_metallic(self, iteration: int) -> bool:
        if iteration < self.albedo_phase_end:
            return False
        return (
            iteration - self.albedo_phase_end
        ) % self.roughness_metallic_interval == 0

    def should_run_albedo_spsa(self, iteration: int) -> bool:
        """Per-material SPSA for albedo — runs every iteration since it's
        the only albedo gradient source (analytical gradient disabled)."""
        if iteration < self.emission_phase_end:
            return False
        return True

    def get_next_spsa_material(self) -> int:
        """Rotate through materials for per-material SPSA."""
        mat = self._spsa_material_cursor
        self._spsa_material_cursor = (
            self._spsa_material_cursor + 1
        ) % self.num_materials
        return mat

    def get_spp(self, iteration: int, current_loss: float = None) -> int:
        if iteration >= self.spp_ramp_end:
            return self.base_spp
        t = iteration / max(self.spp_ramp_end, 1)
        spp = int(self.min_spp + (self.base_spp - self.min_spp) * t)
        return max(2, (spp // 2) * 2)

    def get_views_to_render(self, iteration: int) -> list:
        if self.num_views <= 2:
            return list(range(self.num_views))
        if iteration % self.full_view_interval == 0:
            return list(range(self.num_views))
        n = self.min_views_per_iter
        return sorted(np.random.choice(self.num_views, size=n, replace=False).tolist())

    def get_lr(self, iteration: int) -> float:
        if iteration < self.warmup_iters:
            return self.base_lr * (iteration + 1) / self.warmup_iters
        progress = (iteration - self.warmup_iters) / max(
            self.num_iters - self.warmup_iters, 1
        )
        progress = min(progress, 1.0)
        decay = self.min_lr_factor + (1.0 - self.min_lr_factor) * 0.5 * (
            1.0 + np.cos(np.pi * progress)
        )
        return self.base_lr * decay

    def smooth_gradient(self, key: str, grad_value: float) -> float:
        if key not in self._ema_grads:
            self._ema_grads[key] = grad_value
        else:
            alpha = self.grad_ema_alpha
            self._ema_grads[key] = (
                alpha * grad_value + (1.0 - alpha) * self._ema_grads[key]
            )
        return self._ema_grads[key]

    def track_loss(self, iteration: int, loss: float):
        if self.initial_loss is None:
            self.initial_loss = loss
        self.best_loss = min(self.best_loss, loss)

    def reset(self):
        self.initial_loss = None
        self.best_loss = float("inf")
        self._ema_grads.clear()
        self._spsa_material_cursor = 0

    def status_string(self, iteration: int) -> str:
        spp = self.get_spp(iteration)
        lr = self.get_lr(iteration)
        phase = self.get_phase(iteration)
        return f"SPP:{spp} LR:{lr:.4f} phase:{phase}"
