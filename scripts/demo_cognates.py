"""Visualize cognate divergence between sister lineages.

Two experiments:

1. **Validation against theory.** Simulate ``R`` independent pairs of lineages
   diverging from a shared seed for ``T`` years. Expected shared-cognate
   fraction is ``exp(-2*lam*t)``. We overlay simulation against theory.

2. **Effect of contact.** Repeat with several borrowing rates to show how
   continuous contact pulls the divergence curve upward — the wave component
   of language change.

Saves ``figs/demo_divergence.png``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pie_sim import CognateEvolver, cognate_distance


def simulate_pair(
    lam: float,
    mu: float,
    T_years: int,
    n_meanings: int,
    n_replicates: int,
    seed: int,
    sample_every: int = 25,
) -> tuple[np.ndarray, np.ndarray]:
    """Run ``n_replicates`` paired-lineage simulations.

    Returns:
        ``(sample_times, shared)`` where ``shared[r, k]`` is the fraction of
        cognates shared between the two sisters in replicate ``r`` at time
        ``sample_times[k]``.
    """
    sample_times = np.arange(0, T_years + 1, sample_every)
    if sample_times[-1] != T_years:
        sample_times = np.append(sample_times, T_years)
    sample_set = set(sample_times.tolist())

    rng = np.random.default_rng(seed)
    shared = np.zeros((n_replicates, len(sample_times)))

    for r in range(n_replicates):
        ev = CognateEvolver(
            n_meanings=n_meanings,
            innovation_rate=lam,
            borrowing_rate=mu,
            rng=rng,
        )
        seed_lex = ev.initial_lexicon()
        a, b = seed_lex.copy(), seed_lex.copy()

        idx = 0
        for t in range(T_years + 1):
            if t in sample_set:
                shared[r, idx] = 1.0 - cognate_distance(a, b)
                idx += 1
            if t == T_years:
                break
            if mu > 0:
                a_next = ev.step(a, dt=1.0, neighbors=[b], contact_weights=np.array([1.0]))
                b_next = ev.step(b, dt=1.0, neighbors=[a], contact_weights=np.array([1.0]))
                a, b = a_next, b_next
            else:
                a = ev.step(a, dt=1.0)
                b = ev.step(b, dt=1.0)

    return sample_times, shared


def main() -> None:
    LAMBDA = 1.5e-4
    T = 5000
    M = 200
    R = 20

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ---- Panel 1: isolated lineages vs. theory ----
    ax = axes[0]
    times, shared = simulate_pair(LAMBDA, mu=0.0, T_years=T, n_meanings=M,
                                  n_replicates=R, seed=1)
    mean = shared.mean(axis=0)
    std = shared.std(axis=0)
    theory = np.exp(-2 * LAMBDA * times)

    ax.fill_between(times, mean - std, mean + std, alpha=0.25, color="C0",
                    label=f"simulation $\\pm 1$ SD ($R={R}$)")
    ax.plot(times, mean, color="C0", lw=2, label="simulation mean")
    ax.plot(times, theory, "k--", lw=1.5, label=r"theory: $e^{-2\lambda t}$")
    ax.axvline(5000, color="grey", lw=0.7, ls=":")
    ax.text(5000, 0.95, " ~PIE depth", color="grey", fontsize=9, va="top")
    ax.set_xlabel("years since split")
    ax.set_ylabel("fraction of cognates shared")
    ax.set_title(f"Isolated sisters ($\\lambda = {LAMBDA:g}$/yr, $M = {M}$ meanings)")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)

    # ---- Panel 2: effect of contact ----
    ax = axes[1]
    contact_ratios = [0.0, 0.25, 0.5, 1.0]
    cmap = plt.cm.viridis(np.linspace(0.15, 0.85, len(contact_ratios)))
    for color, ratio in zip(cmap, contact_ratios):
        mu = ratio * LAMBDA
        _, shared = simulate_pair(LAMBDA, mu=mu, T_years=T, n_meanings=M,
                                  n_replicates=R, seed=100 + int(ratio * 100))
        ax.plot(times, shared.mean(axis=0), lw=2, color=color,
                label=fr"$\mu/\lambda = {ratio:.2f}$")
    ax.plot(times, theory, "k--", lw=1.0, alpha=0.6, label=r"isolation theory")
    ax.set_xlabel("years since split")
    ax.set_ylabel("fraction of cognates shared")
    ax.set_title("Continuous contact retards divergence")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)

    fig.suptitle("Cognate divergence under stochastic Dollo + borrowing",
                 fontsize=14, y=1.02)
    fig.tight_layout()

    out_path = Path(__file__).resolve().parents[1] / "figs" / "demo_divergence.png"
    out_path.parent.mkdir(exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
