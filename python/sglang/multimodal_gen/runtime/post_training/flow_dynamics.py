# SPDX-License-Identifier: Apache-2.0
"""Flow-matching reverse-step dynamics for RL rollout / train alignment.

Each concrete class implements **one** reverse step: given velocity ``v`` (model
output), latent ``x`` at ``σ``, and next ``σ'``, return the step mean and noise
scale.  Stochastic sampling is always::

    x' = mean + ε * noise_std          (ε ~ N(0, I))

Log-probability on the full (pre-shard) noise buffer::

    log_prob_no_const = -((ε * noise_std) ** 2)

Step *selection* (which denoise indices are stochastic vs forced ODE) lives in
:func:`resolve_effective_dynamics_type` — not in these classes.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

_LOG_SQRT_2PI = math.log(math.sqrt(2 * math.pi))

CANONICAL_DYNAMICS_TYPES = ("sde", "cps", "ode", "dance_sde")


@dataclass(frozen=True)
class FlowStepCoeffs:
    """One reverse step before drawing ε."""

    prev_sample_mean: torch.Tensor
    noise_std_dev: torch.Tensor
    use_fp32_inputs: bool
    is_deterministic: bool


class FlowDynamicsStep(ABC):
    """Base class for a single flow-matching reverse dynamics."""

    name: str
    default_log_prob_no_const: bool
    supported_on_rollout: bool

    @abstractmethod
    def step_coeffs(
        self,
        model_output: torch.Tensor,
        sample: torch.Tensor,
        current_sigma: torch.Tensor,
        next_sigma: torch.Tensor,
        *,
        noise_level: float,
        sigma_max: float,
    ) -> FlowStepCoeffs:
        """Return ``(prev_sample_mean, noise_std)``; caller draws ε and samples."""

    def aggregate_log_prob_sum(
        self,
        log_prob_no_const_val: torch.Tensor,
        noise_std_dev: torch.Tensor,
        *,
        log_prob_no_const: bool,
    ) -> torch.Tensor:
        """Reduce per-element log-prob tensor to per-batch sums."""
        reduce_dims = list(range(1, log_prob_no_const_val.ndim))
        if log_prob_no_const or self.name == "ode":
            return log_prob_no_const_val.sum(dim=reduce_dims)
        return (
            log_prob_no_const_val / (2 * (noise_std_dev**2))
            - torch.log(noise_std_dev)
            - _LOG_SQRT_2PI
        ).sum(dim=reduce_dims)


class SD3SdeStep(FlowDynamicsStep):
    """SD3 / FlowGRPO standard SDE.

    Used by diffusers SD3 rollout (``rollout_sde_type="sde"``).

    .. math::

        \\sigma_t = \\sqrt{\\frac{\\sigma}{1-\\sigma'}} \\cdot \\eta

        \\text{mean} = x \\left(1 + \\frac{\\sigma_t^2}{2\\sigma} \\Delta\\sigma\\right)
                    + v \\left(1 + \\frac{\\sigma_t^2(1-\\sigma)}{2\\sigma}\\right) \\Delta\\sigma

        \\text{noise\\_std} = \\sigma_t \\sqrt{-\\Delta\\sigma}

    Full Gaussian log-prob (with normalization constant).
    """

    name = "sde"
    default_log_prob_no_const = False
    supported_on_rollout = True

    def step_coeffs(
        self,
        model_output: torch.Tensor,
        sample: torch.Tensor,
        current_sigma: torch.Tensor,
        next_sigma: torch.Tensor,
        *,
        noise_level: float,
        sigma_max: float,
    ) -> FlowStepCoeffs:
        model_output = model_output.float()
        sample = sample.float()
        current_sigma = current_sigma.float()
        next_sigma = next_sigma.float()
        dt = next_sigma - current_sigma

        std_dev_t = (
            torch.sqrt(
                current_sigma
                / (
                    1
                    - torch.where(
                        torch.isclose(current_sigma, current_sigma.new_tensor(1.0)),
                        torch.as_tensor(sigma_max, device=current_sigma.device),
                        current_sigma,
                    )
                )
            )
            * noise_level
        )
        noise_std_dev = std_dev_t * torch.sqrt(-1 * dt)
        prev_sample_mean = (
            sample * (1 + std_dev_t**2 / (2 * current_sigma) * dt)
            + model_output
            * (1 + std_dev_t**2 * (1 - current_sigma) / (2 * current_sigma))
            * dt
        )
        return FlowStepCoeffs(
            prev_sample_mean=prev_sample_mean,
            noise_std_dev=noise_std_dev,
            use_fp32_inputs=True,
            is_deterministic=False,
        )


class CpsStep(FlowDynamicsStep):
    """Coefficients-Preserving Sampling (CPS).

    .. math::

        \\sigma_t = \\sigma' \\sin(\\eta \\pi / 2)

        x_0 = x - \\sigma v, \\quad x_1 = x + v(1-\\sigma)

        \\text{mean} = x_0 (1-\\sigma') + x_1 \\sqrt{\\sigma'^2 - \\sigma_t^2}

        \\text{noise\\_std} = \\sigma_t

    Log-prob without Gaussian normalization constant (``-(x'-\\text{mean})^2``).
    """

    name = "cps"
    default_log_prob_no_const = True
    supported_on_rollout = True

    def step_coeffs(
        self,
        model_output: torch.Tensor,
        sample: torch.Tensor,
        current_sigma: torch.Tensor,
        next_sigma: torch.Tensor,
        *,
        noise_level: float,
        sigma_max: float,
    ) -> FlowStepCoeffs:
        del sigma_max
        model_output = model_output.float()
        sample = sample.float()
        current_sigma = current_sigma.float()
        next_sigma = next_sigma.float()

        std_dev_t = next_sigma * math.sin(noise_level * math.pi / 2)
        noise_std_dev = std_dev_t
        pred_original_sample = sample - current_sigma * model_output
        noise_estimate = sample + model_output * (1 - current_sigma)
        prev_sample_mean = pred_original_sample * (
            1 - next_sigma
        ) + noise_estimate * torch.sqrt(next_sigma**2 - std_dev_t**2)

        return FlowStepCoeffs(
            prev_sample_mean=prev_sample_mean,
            noise_std_dev=noise_std_dev,
            use_fp32_inputs=True,
            is_deterministic=False,
        )


class OdeStep(FlowDynamicsStep):
    """Deterministic Euler step (no diffusion noise).

    .. math::

        x' = x + \\Delta\\sigma \\cdot v

    Log-prob is zero; callers should set ``rollout_log_prob_no_const=True``.
    """

    name = "ode"
    default_log_prob_no_const = True
    supported_on_rollout = True

    def step_coeffs(
        self,
        model_output: torch.Tensor,
        sample: torch.Tensor,
        current_sigma: torch.Tensor,
        next_sigma: torch.Tensor,
        *,
        noise_level: float,
        sigma_max: float,
    ) -> FlowStepCoeffs:
        del noise_level, sigma_max
        dt = next_sigma - current_sigma
        prev_sample_mean = sample + dt * model_output
        noise_std_dev = torch.zeros(
            (), device=model_output.device, dtype=model_output.dtype
        )
        return FlowStepCoeffs(
            prev_sample_mean=prev_sample_mean,
            noise_std_dev=noise_std_dev,
            use_fp32_inputs=False,
            is_deterministic=True,
        )


class DanceSdeStep(FlowDynamicsStep):
    """Dance-SDE variant (train-side recompute; rollout not wired yet).

    .. math::

        x_0 = x - \\sigma v

        \\text{log\\_term} = \\frac{\\eta^2}{2} \\frac{x - x_0(1-\\sigma)}{\\sigma^2}

        \\text{mean} = x + (v + \\text{log\\_term}) \\Delta\\sigma

        \\text{noise\\_std} = \\eta \\sqrt{-\\Delta\\sigma}

    Full Gaussian log-prob.
    """

    name = "dance_sde"
    default_log_prob_no_const = False
    supported_on_rollout = False

    def step_coeffs(
        self,
        model_output: torch.Tensor,
        sample: torch.Tensor,
        current_sigma: torch.Tensor,
        next_sigma: torch.Tensor,
        *,
        noise_level: float,
        sigma_max: float,
    ) -> FlowStepCoeffs:
        del sigma_max
        model_output = model_output.float()
        sample = sample.float()
        current_sigma = current_sigma.float()
        next_sigma = next_sigma.float()
        dt = next_sigma - current_sigma

        sigma_safe = torch.clamp(current_sigma, min=1e-8)
        x0_pred = sample - sigma_safe * model_output
        std_dev_t = torch.as_tensor(
            noise_level, dtype=sample.dtype, device=sample.device
        )
        log_term = (
            0.5
            * noise_level**2
            * (sample - x0_pred * (1.0 - current_sigma))
            / (sigma_safe**2)
        )
        prev_sample_mean = sample + (model_output + log_term) * dt
        noise_std_dev = std_dev_t * torch.sqrt(torch.clamp(-dt, min=1e-12))

        return FlowStepCoeffs(
            prev_sample_mean=prev_sample_mean,
            noise_std_dev=noise_std_dev,
            use_fp32_inputs=True,
            is_deterministic=False,
        )


# Singleton instances — import and call ``CPS.step_coeffs(...)`` directly if helpful.
SD3_SDE = SD3SdeStep()
CPS = CpsStep()
ODE = OdeStep()
DANCE_SDE = DanceSdeStep()

FLOW_DYNAMICS_BY_NAME: dict[str, FlowDynamicsStep] = {
    SD3_SDE.name: SD3_SDE,
    CPS.name: CPS,
    ODE.name: ODE,
    DANCE_SDE.name: DANCE_SDE,
}

ROLLOUT_DYNAMICS_TYPES = tuple(
    name
    for name, step in FLOW_DYNAMICS_BY_NAME.items()
    if step.supported_on_rollout
)


def normalize_dynamics_type(name: str) -> str:
    """Map CLI / legacy aliases (``CPS``, …) to canonical names."""
    key = str(name).strip().lower().replace("-", "_")
    if key not in FLOW_DYNAMICS_BY_NAME:
        raise ValueError(
            f"Unknown dynamics_type {name!r}; expected one of {CANONICAL_DYNAMICS_TYPES}"
        )
    return key


def get_flow_dynamics(name: str) -> FlowDynamicsStep:
    """Look up the dynamics implementation for ``name``."""
    return FLOW_DYNAMICS_BY_NAME[normalize_dynamics_type(name)]


def resolve_effective_dynamics_type(
    dynamics_type: str,
    *,
    loop_step_index: int | None,
    sde_step_indices: list[int] | None,
) -> str:
    """Steps outside ``sde_step_indices`` are forced to ODE."""
    dynamics_type = normalize_dynamics_type(dynamics_type)
    if (
        dynamics_type != "ode"
        and sde_step_indices is not None
        and loop_step_index is not None
        and loop_step_index not in sde_step_indices
    ):
        return "ode"
    return dynamics_type


def compute_flow_step_coeffs(
    dynamics_type: str,
    model_output: torch.Tensor,
    sample: torch.Tensor,
    current_sigma: torch.Tensor,
    next_sigma: torch.Tensor,
    *,
    noise_level: float,
    sigma_max: float,
) -> FlowStepCoeffs:
    """Dispatch to the concrete dynamics class for ``dynamics_type``."""
    return get_flow_dynamics(dynamics_type).step_coeffs(
        model_output,
        sample,
        current_sigma,
        next_sigma,
        noise_level=noise_level,
        sigma_max=sigma_max,
    )


def log_prob_no_const_from_noise(
    full_variance_noise: torch.Tensor,
    noise_std_dev: torch.Tensor,
) -> torch.Tensor:
    return -((full_variance_noise * noise_std_dev) ** 2)


def aggregate_log_prob_sum(
    log_prob_no_const_val: torch.Tensor,
    noise_std_dev: torch.Tensor,
    *,
    log_prob_no_const: bool,
    dynamics_type: str,
) -> torch.Tensor:
    return get_flow_dynamics(dynamics_type).aggregate_log_prob_sum(
        log_prob_no_const_val,
        noise_std_dev,
        log_prob_no_const=log_prob_no_const,
    )


def validate_rollout_noise_level(
    noise_level: float,
    dynamics_type: str,
    *,
    log_prob_no_const: bool,
) -> None:
    dynamics = get_flow_dynamics(dynamics_type)
    if not log_prob_no_const and not dynamics.name == "ode":
        if noise_level <= 0:
            raise AssertionError(
                "True log-probability computation requires a non-zero noise level."
            )


def validate_ode_log_prob_config(
    requested_dynamics_type: str,
    effective_dynamics_type: str,
    *,
    log_prob_no_const: bool,
) -> None:
    if (
        normalize_dynamics_type(requested_dynamics_type) == "ode"
        and normalize_dynamics_type(effective_dynamics_type) == "ode"
        and not log_prob_no_const
    ):
        raise AssertionError(
            "p_ode is always 0, true log_prob is meaningless, set "
            "rollout_log_prob_no_const to True to enable log_prob computation"
        )
