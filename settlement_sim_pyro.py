"""settlement_sim_pyro.py
=======================
A fully probabilistic reformulation of the AstarWorld settlement simulation
as a Pyro computational graph.

Design
------
Rather than running N separate Monte-Carlo trajectories and averaging, this
module models the *distribution* over grid states directly:

  * Each cell (h, w) carries a probability vector over NUM_CLASSES = 6 classes.
  * Global dynamics parameters (growth rates, collapse rates, etc.) are treated
    as latent random variables with weakly-informative priors matched to the
    original SimParams defaults.
  * At each simulated year a differentiable "soft step" propagates the
    probability tensor forward, using 2-D depthwise convolutions to accumulate
    neighbourhood statistics.
  * The final output is a Dirichlet-distributed random variable whose
    concentration is proportional to the terminal soft-grid, giving a proper
    probability distribution over output grids.

Class index mapping (matches the original API ground-truth format)
------------------------------------------------------------------
  0 – BG       (ocean / empty / plains)
  1 – SETTLE   (active settlement)
  2 – PORT     (settlement with port)
  3 – RUIN     (collapsed settlement)
  4 – FOREST
  5 – MOUNTAIN

Usage
-----
    from settlement_sim_pyro import (
        make_initial_grid,
        sample_prior,
        predictive_mean,
        run_svi,
        settlement_model,
        settlement_guide,
    )

    # Build a soft initial grid from a freshly seeded AstarWorld
    initial = make_initial_grid(seed=42)            # (40, 40, 6)

    # Draw 200 forward samples from the prior predictive distribution
    samples = sample_prior(initial, years=50, n_samples=200)
    # → (200, 40, 40, 6) ; mean over axis-0 ≡ original run_monte_carlo output

    mean_pred = predictive_mean(initial, years=50, n_samples=200)
    # → (40, 40, 6)

    # Fit posterior to an observed output grid via SVI
    posterior_params = run_svi(initial, observed_grid, years=50, n_steps=1000)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import ClippedAdam
import numpy as np
from typing import Optional

import settlement_sim as _base


# ─────────────────────────────────────────────────────────────────────────────
# Class-index constants
# ─────────────────────────────────────────────────────────────────────────────

CLS_BG       = 0   # ocean / empty / plains
CLS_SETTLE   = 1
CLS_PORT     = 2
CLS_RUIN     = 3
CLS_FOREST   = 4
CLS_MOUNTAIN = 5
NUM_CLASSES  = 6

# Maps the original integer terrain codes to the 6-class scheme
_CLASS_MAP: dict[int, int] = {
    _base.OCEAN:    CLS_BG,
    _base.PLAINS:   CLS_BG,
    _base.EMPTY:    CLS_BG,
    _base.SETTLE:   CLS_SETTLE,
    _base.PORT:     CLS_PORT,
    _base.RUIN:     CLS_RUIN,
    _base.FOREST:   CLS_FOREST,
    _base.MOUNTAIN: CLS_MOUNTAIN,
}


# ─────────────────────────────────────────────────────────────────────────────
# Grid-construction helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_initial_grid(
    seed:   int  = 0,
    width:  int  = 40,
    height: int  = 40,
    noise:  float = 0.02,
) -> torch.Tensor:
    """
    Build a soft-probability initial grid from a freshly generated AstarWorld.

    Args:
        seed:          RNG seed passed to AstarWorld map generation.
        width, height: Grid dimensions (default 40×40).
        noise:         Small uniform noise added before normalisation so that
                       every class has a nonzero prior probability.

    Returns:
        Float tensor of shape (height, width, NUM_CLASSES) with rows summing
        to 1.  Mountain and ocean cells have near-certain class assignments;
        all other cells carry a small background uncertainty.
    """
    params = _base.SimParams()
    world  = _base.AstarWorld(seed=seed, params=params, width=width, height=height)

    hard = np.zeros((height, width, NUM_CLASSES), dtype=np.float32)
    for y in range(height):
        for x in range(width):
            cls = _CLASS_MAP.get(int(world.grid[y, x]), CLS_BG)
            hard[y, x, cls] = 1.0

    grid = torch.tensor(hard)
    if noise > 0.0:
        grid = grid + noise / NUM_CLASSES
    grid = grid / grid.sum(dim=-1, keepdim=True)
    return grid


def grid_from_numpy(arr: np.ndarray, noise: float = 0.0) -> torch.Tensor:
    """
    Convert a (H, W, C) NumPy probability array to a normalised Tensor.

    Useful for wrapping the output of the original ``run_monte_carlo`` before
    passing it as an observation to :func:`run_svi`.
    """
    t = torch.tensor(arr, dtype=torch.float32)
    if noise > 0.0:
        t = t + noise / NUM_CLASSES
    t = t / t.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    return t


# ─────────────────────────────────────────────────────────────────────────────
# Differentiable simulation core
# ─────────────────────────────────────────────────────────────────────────────

def _neighborhood_counts(grid: torch.Tensor) -> torch.Tensor:
    """
    Compute the expected number of each class among the 8 (Chebyshev)
    neighbours of every cell via a differentiable depthwise 2-D convolution.

    Args:
        grid: (H, W, NUM_CLASSES) probability tensor.

    Returns:
        (H, W, NUM_CLASSES) — for each cell, the sum of its neighbours'
        class-probability vectors.  Values lie in [0, 8].
    """
    # (H, W, C) → (1, C, H, W) for F.conv2d
    x = grid.permute(2, 0, 1).unsqueeze(0)
    C = x.shape[1]

    # 3×3 all-ones kernel; zero at centre → 8-neighbour sum
    kernel = torch.ones(1, 1, 3, 3, device=grid.device, dtype=grid.dtype)
    kernel[0, 0, 1, 1] = 0.0

    # Depthwise convolution: each channel convolved with the same kernel
    kernel_dw = kernel.expand(C, 1, 3, 3).contiguous()          # (C, 1, 3, 3)
    out = F.conv2d(x, kernel_dw, padding=1, groups=C)            # (1, C, H, W)
    return out.squeeze(0).permute(1, 2, 0)                       # (H, W, C)


def _soft_step(
    grid:              torch.Tensor,
    expansion_prob:    torch.Tensor,
    port_build_prob:   torch.Tensor,
    collapse_rate:     torch.Tensor,
    reclaim_prob:      torch.Tensor,
    forest_grow_prob:  torch.Tensor,
    raid_success_rate: torch.Tensor,
    winter_kills:      torch.Tensor,
) -> torch.Tensor:
    """
    One year of soft probabilistic transitions on the class-probability grid.

    All operations are differentiable with respect to both the grid tensor and
    the scalar parameter tensors, enabling gradient-based inference over the
    dynamics parameters.

    Transition rules
    ----------------
    BG       → SETTLE   expansion driven by neighbouring settlements
    BG       → FOREST   forest encroachment from adjacent forest cells
    FOREST   → SETTLE   clearing for new colonies
    SETTLE   → PORT     port development for coastal settlements
    SETTLE   → RUIN     collapse from raids and/or winter starvation
    PORT     → RUIN     same collapse pathway
    RUIN     → SETTLE   reclamation by a nearby patron settlement
    RUIN     → FOREST   natural succession after prolonged abandonment
    MOUNTAIN → MOUNTAIN static (no transitions)

    Args:
        grid:              (H, W, NUM_CLASSES) current probability distribution.
        expansion_prob:    Scalar — probability a BG cell is colonised per year
                           when surrounded by ~3 settlement neighbours.
        port_build_prob:   Scalar — annual probability that a coastal settlement
                           develops a port.
        collapse_rate:     Scalar — baseline annual collapse probability for
                           any active settlement.
        reclaim_prob:      Scalar — annual probability that a ruin adjacent to
                           settlements is reclaimed.
        forest_grow_prob:  Scalar — annual probability of forest regeneration
                           in ruined or bare cells.
        raid_success_rate: Scalar — fraction of raids that succeed, increasing
                           the effective collapse pressure on target settlements.
        winter_kills:      Scalar — extra collapse probability contribution from
                           winter starvation events (pre-computed from severity).

    Returns:
        (H, W, NUM_CLASSES) updated probability tensor, rows sum to 1.
    """
    nbr = _neighborhood_counts(grid)                              # (H, W, C)

    nbr_settle       = nbr[..., CLS_SETTLE]
    nbr_port         = nbr[..., CLS_PORT]
    nbr_forest       = nbr[..., CLS_FOREST]
    nbr_settle_total = nbr_settle + nbr_port                      # combined pressure

    # Current class-probability slices
    p_bg = grid[..., CLS_BG]
    p_s  = grid[..., CLS_SETTLE]
    p_p  = grid[..., CLS_PORT]
    p_r  = grid[..., CLS_RUIN]
    p_f  = grid[..., CLS_FOREST]
    p_m  = grid[..., CLS_MOUNTAIN]

    # ── BG transitions ──────────────────────────────────────────────────────

    # BG → SETTLE: logistic pressure so probability saturates around 3 neighbours
    expand_rate = torch.sigmoid(nbr_settle_total - 3.0) * expansion_prob
    p_bg_to_s   = torch.minimum(p_bg * expand_rate, p_bg).clamp(min=0.0)

    # BG → FOREST: slow encroachment when adjacent to existing forest
    forest_rate = torch.sigmoid(nbr_forest - 2.0) * forest_grow_prob * 0.3
    p_bg_to_f   = torch.minimum(p_bg * forest_rate, (p_bg - p_bg_to_s).clamp(min=0.0)).clamp(min=0.0)

    new_p_bg = (p_bg - p_bg_to_s - p_bg_to_f).clamp(min=0.0)

    # ── FOREST transitions ───────────────────────────────────────────────────

    # FOREST → SETTLE: clearing land for colonisation (lower rate than BG)
    clearing_rate = nbr_settle_total * expansion_prob * 0.2
    p_f_to_s      = torch.minimum(p_f * clearing_rate, p_f).clamp(min=0.0)
    new_p_f       = (p_f - p_f_to_s).clamp(min=0.0)

    # ── SETTLE transitions ───────────────────────────────────────────────────

    # SETTLE → PORT: coastal only (fraction of BG neighbours as proxy for coast)
    coastal   = (nbr[..., CLS_BG] / 8.0).clamp(0.0, 1.0)
    p_s_to_p  = torch.minimum(p_s * coastal * port_build_prob * 0.25, p_s).clamp(min=0.0)

    # SETTLE → RUIN: raids + winter starvation
    raid_pressure = (nbr_settle_total * raid_success_rate * 0.1).clamp(0.0, 0.3)
    total_collapse_s = (collapse_rate + winter_kills + raid_pressure).clamp(0.0, 1.0)
    p_s_collapse = torch.minimum(
        p_s * total_collapse_s, (p_s - p_s_to_p).clamp(min=0.0)
    ).clamp(min=0.0)

    new_p_s = (
        p_s - p_s_to_p - p_s_collapse
        + p_bg_to_s + p_f_to_s
    ).clamp(min=0.0)

    # ── PORT transitions ─────────────────────────────────────────────────────

    total_collapse_p = (collapse_rate + winter_kills).clamp(0.0, 1.0)
    p_p_collapse = torch.minimum(p_p * total_collapse_p, p_p).clamp(min=0.0)
    new_p_p      = (p_p - p_p_collapse + p_s_to_p).clamp(min=0.0)

    # ── RUIN transitions ─────────────────────────────────────────────────────

    ruin_in = p_s_collapse + p_p_collapse

    # RUIN → SETTLE: reclamation by nearby patron
    patron_pressure = nbr_settle_total * reclaim_prob * 0.2
    p_r_to_s = torch.minimum(p_r * patron_pressure, p_r).clamp(min=0.0)

    # RUIN → FOREST: natural succession
    p_r_to_f = torch.minimum(
        p_r * forest_grow_prob * 0.5, (p_r - p_r_to_s).clamp(min=0.0)
    ).clamp(min=0.0)

    new_p_r = (p_r - p_r_to_s - p_r_to_f + ruin_in).clamp(0.0)

    # Reclaimed ruins rejoin settlement mass
    new_p_s = (new_p_s + p_r_to_s).clamp(0.0)

    # Forest gains
    new_p_f = (new_p_f + p_r_to_f + p_bg_to_f).clamp(0.0)

    # ── MOUNTAIN: static ────────────────────────────────────────────────────
    new_p_m = p_m

    # ── Assemble and renormalise ─────────────────────────────────────────────
    new_grid = torch.stack(
        [new_p_bg, new_p_s, new_p_p, new_p_r, new_p_f, new_p_m], dim=-1
    )
    total = new_grid.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    return new_grid / total


# ─────────────────────────────────────────────────────────────────────────────
# Pyro model and guide
# ─────────────────────────────────────────────────────────────────────────────

def settlement_model(
    initial_grid: torch.Tensor,
    years: int = 50,
    obs: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Pyro probabilistic model for Viking-era settlement evolution on a 40×40 grid.

    Latent variables (all scalar, sampled once per forward pass)
    ------------------------------------------------------------
    expansion_prob    ~ Beta(1.5, 13.5)  mean ≈ 0.10
        Annual probability that a background cell colonises, given settlement
        neighbours.

    port_build_prob   ~ Beta(2.0, 8.0)   mean ≈ 0.20
        Annual probability that a coastal settlement develops a port.

    collapse_rate     ~ Beta(1.0, 9.0)   mean ≈ 0.10
        Baseline annual probability that any active settlement collapses.

    reclaim_prob      ~ Beta(1.5, 8.5)   mean ≈ 0.15
        Annual probability that an abandoned ruin is reclaimed when adjacent
        to thriving settlements.

    forest_grow_prob  ~ Beta(1.0, 19.0)  mean ≈ 0.05
        Annual probability of natural vegetation re-establishment in ruins
        or bare cells.

    raid_success_rate ~ Beta(2.0, 6.0)   mean ≈ 0.25
        Fraction of raids that succeed, scaling extra collapse pressure.

    winter_severity   ~ LogNormal(0, 0.6) mean ≈ 1.21
        Annual winter harshness; a heavy-tailed distribution naturally captures
        rare harsh-winter spikes (equivalent to the original Bernoulli spike
        mechanism but fully continuous for gradient-based inference).

    Likelihood
    ----------
    For each cell (h, w), the output is modelled as:

        output[h, w] ~ Dirichlet(α · final_grid[h, w] + 1)

    where α = 5.0 and the floor of 1.0 keeps log-probabilities finite for
    observations at the simplex boundary (e.g. hard one-hot grids).

    Args:
        initial_grid: (H, W, NUM_CLASSES) float tensor of initial class
                      probability distributions; rows must sum to 1.
        years:        Number of simulation years to run (default 50).
        obs:          Optional (H, W, NUM_CLASSES) observed output to condition
                      on.  When provided, the model computes a likelihood and
                      can be used for posterior inference via SVI or MCMC.

    Returns:
        (H, W, NUM_CLASSES) sample from the output Dirichlet distribution.
    """
    H, W, _ = initial_grid.shape

    # ── Sample dynamics parameters ───────────────────────────────────────────
    expansion_prob    = pyro.sample("expansion_prob",    dist.Beta(1.5, 13.5))
    port_build_prob   = pyro.sample("port_build_prob",   dist.Beta(2.0,  8.0))
    collapse_rate     = pyro.sample("collapse_rate",     dist.Beta(1.0,  9.0))
    reclaim_prob      = pyro.sample("reclaim_prob",      dist.Beta(1.5,  8.5))
    forest_grow_prob  = pyro.sample("forest_grow_prob",  dist.Beta(1.0, 19.0))
    raid_success_rate = pyro.sample("raid_success_rate", dist.Beta(2.0,  6.0))

    # Heavy-tailed winter severity; large draws represent harsh-winter years
    winter_severity = pyro.sample("winter_severity", dist.LogNormal(0.0, 0.6))
    winter_kills    = (winter_severity / 10.0).clamp(0.0, 0.4)

    # ── Run the soft probabilistic simulation ────────────────────────────────
    grid = initial_grid
    for _ in range(years):
        grid = _soft_step(
            grid,
            expansion_prob    = expansion_prob,
            port_build_prob   = port_build_prob,
            collapse_rate     = collapse_rate,
            reclaim_prob      = reclaim_prob,
            forest_grow_prob  = forest_grow_prob,
            raid_success_rate = raid_success_rate,
            winter_kills      = winter_kills,
        )

    # ── Output likelihood: independent Dirichlet per cell ────────────────────
    # Concentration: strong peak at the terminal soft distribution.
    # Floor of 1.0 (not < 1) ensures (α_i - 1)*log(x_i) = 0 when x_i = 0,
    # keeping the log-probability finite for observed grids that contain exact
    # zeros (e.g. Monte-Carlo averages with few runs, or hard one-hot grids).
    concentration_scale = 5.0
    concentration = grid * concentration_scale + 1.0               # (H, W, C)

    # dist.Dirichlet(concentration) has batch_shape=(H, W), event_shape=(C,).
    # Wrapping with Independent(…, 2) absorbs the (H, W) batch dims into the
    # event so the entire grid is one sample site with event_shape=(H, W, C).
    # This means pyro.infer.Predictive returns shape (n_samples, H, W, C).
    # When observations are provided, push them into the open simplex interior.
    # This handles hard / sparse grids (e.g. one-hot MC averages) that have
    # exact zeros; the Dirichlet distribution is only defined for x_i > 0.
    if obs is not None:
        obs_safe = (obs + 1e-6) / (obs + 1e-6).sum(dim=-1, keepdim=True)
    else:
        obs_safe = None

    output = pyro.sample(
        "output",
        dist.Independent(dist.Dirichlet(concentration), 2),
        obs=obs_safe,
    )

    return output                                                    # (H, W, C)


def settlement_guide(
    initial_grid: torch.Tensor,
    years:        int  = 50,
    obs:          Optional[torch.Tensor] = None,
) -> None:
    """
    Mean-field variational guide for :func:`settlement_model`.

    Approximates the posterior over all dynamics parameters with independent
    Beta / LogNormal distributions whose natural parameters are stored in
    Pyro's param store and updated by SVI.

    All parameters are initialised to match the model priors so that the guide
    starts at the prior and is refined toward the posterior during optimisation.

    Usage::

        optimizer = ClippedAdam({"lr": 1e-3})
        svi = SVI(settlement_model, settlement_guide, optimizer, Trace_ELBO())
        for step in range(n_steps):
            loss = svi.step(initial_grid, years=50, obs=observed_grid)
    """
    def _beta_site(name: str, alpha0: float, beta0: float) -> None:
        # Store raw (unconstrained) log parameters; exponentiate to get positive α,β
        log_a = pyro.param(f"{name}_log_alpha", torch.tensor(alpha0).log())
        log_b = pyro.param(f"{name}_log_beta",  torch.tensor(beta0).log())
        pyro.sample(name, dist.Beta(log_a.exp().clamp(min=1e-3),
                                    log_b.exp().clamp(min=1e-3)))

    _beta_site("expansion_prob",    1.5, 13.5)
    _beta_site("port_build_prob",   2.0,  8.0)
    _beta_site("collapse_rate",     1.0,  9.0)
    _beta_site("reclaim_prob",      1.5,  8.5)
    _beta_site("forest_grow_prob",  1.0, 19.0)
    _beta_site("raid_success_rate", 2.0,  6.0)

    # LogNormal for winter_severity
    loc      = pyro.param("winter_severity_loc",       torch.tensor(0.0))
    log_scale = pyro.param("winter_severity_log_scale", torch.tensor(0.6).log())
    pyro.sample("winter_severity", dist.LogNormal(loc, log_scale.exp().clamp(min=1e-4)))


# ─────────────────────────────────────────────────────────────────────────────
# Convenient inference / sampling API
# ─────────────────────────────────────────────────────────────────────────────

def sample_prior(
    initial_grid: torch.Tensor,
    years:        int = 50,
    n_samples:    int = 200,
) -> torch.Tensor:
    """
    Draw forward samples from the prior predictive distribution.

    This is the probabilistic-graph equivalent of ``run_monte_carlo()`` in the
    original module: instead of running N independent deterministic simulations
    and averaging, it samples the dynamics parameters from their priors and
    propagates forward through the shared computational graph.

    Args:
        initial_grid: (H, W, NUM_CLASSES) initial probability tensor.
        years:        Number of simulation years.
        n_samples:    Number of independent prior samples.

    Returns:
        (n_samples, H, W, NUM_CLASSES) tensor.
        ``samples.mean(0)`` gives the prior predictive mean, equivalent to
        the Monte-Carlo average produced by ``run_monte_carlo()``.
    """
    predictive = pyro.infer.Predictive(settlement_model, num_samples=n_samples)
    samples    = predictive(initial_grid, years=years)
    return samples["output"]                                     # (N, H, W, C)


def predictive_mean(
    initial_grid: torch.Tensor,
    years:        int = 50,
    n_samples:    int = 200,
) -> torch.Tensor:
    """
    Compute the prior-predictive mean over output grids.

    Drop-in replacement for ``run_monte_carlo()`` that operates entirely
    through the Pyro computational graph.

    Returns:
        (H, W, NUM_CLASSES) float tensor with rows summing to 1.
    """
    return sample_prior(initial_grid, years=years, n_samples=n_samples).mean(0)


def run_svi(
    initial_grid:  torch.Tensor,
    observed_grid: torch.Tensor,
    years:         int   = 50,
    n_steps:       int   = 1000,
    lr:            float = 1e-3,
    verbose:       bool  = True,
) -> dict:
    """
    Fit the variational guide to the posterior via Stochastic Variational
    Inference, conditioning on an observed output grid.

    After completion the Pyro param store holds the optimised variational
    parameters.  You can draw posterior-predictive samples via::

        guide_trace = pyro.infer.Predictive(
            settlement_model,
            guide=settlement_guide,
            num_samples=200,
        )(initial_grid, years=years)

    Args:
        initial_grid:  (H, W, NUM_CLASSES) initial state.
        observed_grid: (H, W, NUM_CLASSES) ground-truth output (e.g. the
                       result of ``run_monte_carlo`` wrapped with
                       ``grid_from_numpy``).
        years:         Number of simulation years.
        n_steps:       SVI gradient steps.
        lr:            Adam learning rate.
        verbose:       Print ELBO every 100 steps.

    Returns:
        dict with keys:
          * ``"losses"``          — list of per-step ELBO values (negated loss)
          * one key per parameter — posterior mean of each dynamics parameter
    """
    pyro.clear_param_store()
    optimizer = ClippedAdam({"lr": lr})
    svi = SVI(
        model = settlement_model,
        guide = settlement_guide,
        optim = optimizer,
        loss  = Trace_ELBO(),
    )

    losses: list[float] = []
    for step in range(n_steps):
        loss = svi.step(initial_grid, years=years, obs=observed_grid)
        losses.append(loss)
        if verbose and (step + 1) % 100 == 0:
            print(f"  [SVI {step+1:5d}/{n_steps}]  ELBO = {-loss:,.1f}")

    # Collect posterior means from the param store
    ps      = pyro.get_param_store()
    results: dict = {"losses": losses}

    param_names = set(ps.get_all_param_names())

    for name in [
        "expansion_prob", "port_build_prob", "collapse_rate",
        "reclaim_prob", "forest_grow_prob", "raid_success_rate",
    ]:
        a_key, b_key = f"{name}_log_alpha", f"{name}_log_beta"
        if a_key in param_names and b_key in param_names:
            a_v = ps[a_key].exp().item()
            b_v = ps[b_key].exp().item()
            results[name] = a_v / (a_v + b_v)            # Beta posterior mean

    if "winter_severity_loc" in param_names and "winter_severity_log_scale" in param_names:
        loc_v     = ps["winter_severity_loc"]
        scale_val = ps["winter_severity_log_scale"].exp()
        results["winter_severity"] = torch.exp(
            loc_v + 0.5 * scale_val ** 2
        ).item()                                          # LogNormal mean

    return results


def posterior_predictive_samples(
    initial_grid: torch.Tensor,
    years:        int = 50,
    n_samples:    int = 200,
) -> torch.Tensor:
    """
    Draw posterior predictive samples after running :func:`run_svi`.

    Call :func:`run_svi` first to optimise the guide parameters; then call
    this function to obtain samples from the posterior predictive distribution
    P(output | observed).

    Args:
        initial_grid: (H, W, NUM_CLASSES)
        years:        Number of simulation years.
        n_samples:    Number of posterior samples.

    Returns:
        (n_samples, H, W, NUM_CLASSES) tensor.
    """
    predictive = pyro.infer.Predictive(
        model      = settlement_model,
        guide      = settlement_guide,
        num_samples = n_samples,
    )
    samples = predictive(initial_grid, years=years)
    return samples["output"]                                     # (N, H, W, C)
