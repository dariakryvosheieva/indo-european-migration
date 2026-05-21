"""Cognate-class evolution under a stochastic-Dollo model with borrowing.

A *lexicon* is represented as a vector of integer cognate-class IDs, one per
meaning slot in a Swadesh-style word list. Two languages share a cognate at
slot ``i`` iff their IDs at ``i`` are equal.

The Dollo property is preserved automatically: every innovation mints a
globally-fresh ID via :class:`CognateEvolver`, so two lineages can never
independently invent the same class.

Two stochastic processes act independently on each slot per generation:

- **Innovation** at per-slot rate ``lam`` — replace the ID with a fresh one.
  Captures lexical replacement (semantic shift, taboo, novel coinage).
- **Borrowing** at per-slot rate ``mu``, weighted by neighbor proximity —
  copy a neighbor's ID. Captures language contact.

Default rate calibration follows glottochronology: ~14% replacement per
millennium for basic vocabulary, giving ``lam = 1.5e-4`` per year per slot.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class Lexicon:
    """A language's lexicon as cognate-class IDs over Swadesh-style meaning slots.

    Attributes:
        cognate_ids: Integer array of shape ``(n_meanings,)``. Entry ``i`` is
            the cognate-class ID currently occupying meaning slot ``i``.
    """

    cognate_ids: np.ndarray

    def __post_init__(self) -> None:
        self.cognate_ids = np.asarray(self.cognate_ids, dtype=np.int64)
        if self.cognate_ids.ndim != 1:
            raise ValueError("cognate_ids must be 1-D")

    @property
    def n_meanings(self) -> int:
        return int(self.cognate_ids.size)

    def copy(self) -> "Lexicon":
        return Lexicon(self.cognate_ids.copy())


class CognateEvolver:
    """Forward-time simulator for cognate evolution.

    Maintains a global cognate-ID counter so that every innovation produces an
    ID never seen before in the simulation, enforcing the Dollo property.

    Args:
        n_meanings: Length of the Swadesh-style word list (default 200).
        innovation_rate: Per-slot per-year probability rate of lexical
            replacement. Default ``1.5e-4`` corresponds to ~14% replacement
            per millennium (Swadesh's classic glottochronological constant).
        borrowing_rate: Per-slot per-year probability rate of contact-induced
            replacement when a neighbor is available. Default 0.
        rng: Optional numpy ``Generator`` for reproducibility.
    """

    def __init__(
        self,
        n_meanings: int = 200,
        innovation_rate: float = 1.5e-4,
        borrowing_rate: float = 0.0,
        rng: np.random.Generator | None = None,
    ) -> None:
        if n_meanings <= 0:
            raise ValueError("n_meanings must be positive")
        if innovation_rate < 0 or borrowing_rate < 0:
            raise ValueError("rates must be non-negative")
        self.n_meanings = int(n_meanings)
        self.lam = float(innovation_rate)
        self.mu = float(borrowing_rate)
        self.rng = rng if rng is not None else np.random.default_rng()
        self._next_id: int = 0

    def _fresh_ids(self, n: int) -> np.ndarray:
        ids = np.arange(self._next_id, self._next_id + n, dtype=np.int64)
        self._next_id += n
        return ids

    def initial_lexicon(self) -> Lexicon:
        """Mint a fresh PIE seed lexicon: one unique cognate ID per meaning slot."""
        return Lexicon(self._fresh_ids(self.n_meanings))

    def step(
        self,
        lex: Lexicon,
        dt: float = 1.0,
        neighbors: Sequence[Lexicon] | None = None,
        contact_weights: np.ndarray | Sequence[float] | None = None,
        borrow_intensity: float = 1.0,
    ) -> Lexicon:
        """Advance ``lex`` by ``dt`` years.

        Per-slot per-year rates are converted to per-step probabilities via
        ``1 - exp(-rate * dt)`` so the model is consistent under coarsening
        of the time grid.

        Args:
            lex: Current lexicon.
            dt: Time step in years.
            neighbors: Optional list of neighbor lexicons that this language
                may borrow from this step.
            contact_weights: Non-negative weights of the same length as
                ``neighbors``, proportional to relative contribution of each
                neighbor (e.g., ``population * exp(-distance / sigma)``).
                They are normalized internally and used only for source
                selection — the *total* borrowing rate is governed by
                ``self.mu * borrow_intensity``.
            borrow_intensity: Multiplier on the base borrowing rate
                ``self.mu`` for this step. Use to encode the strength of
                contact (e.g., total weight of populated neighbors). Default
                ``1.0`` reproduces the rate-as-specified-at-construction
                behavior.

        Returns:
            A new :class:`Lexicon` reflecting one step of evolution.
        """
        if lex.n_meanings != self.n_meanings:
            raise ValueError(
                f"lexicon has {lex.n_meanings} meanings, evolver expects {self.n_meanings}"
            )
        new_ids = lex.cognate_ids.copy()

        p_innov = -np.expm1(-self.lam * dt)  # 1 - exp(-lam*dt), stable for small lam*dt
        innovate = self.rng.random(self.n_meanings) < p_innov
        n_innov = int(innovate.sum())
        if n_innov:
            new_ids[innovate] = self._fresh_ids(n_innov)

        if neighbors and self.mu > 0.0 and contact_weights is not None and borrow_intensity > 0.0:
            w = np.asarray(contact_weights, dtype=np.float64)
            if w.shape[0] != len(neighbors):
                raise ValueError("contact_weights length must match neighbors")
            total = w.sum()
            if total > 0.0:
                p_borrow = -np.expm1(-self.mu * dt * borrow_intensity)
                borrow_mask = self.rng.random(self.n_meanings) < p_borrow
                # Don't double-update slots that just innovated
                borrow_mask &= ~innovate
                if borrow_mask.any():
                    probs = w / total
                    sources = self.rng.choice(len(neighbors), size=int(borrow_mask.sum()), p=probs)
                    slot_idxs = np.flatnonzero(borrow_mask)
                    for slot, src in zip(slot_idxs, sources):
                        new_ids[slot] = neighbors[src].cognate_ids[slot]

        return Lexicon(new_ids)


def cognate_distance(a: Lexicon, b: Lexicon) -> float:
    """Lexical distance: fraction of meaning slots with non-shared cognates.

    Equivalent to ``1 - p_shared`` where ``p_shared`` is the proportion of
    Swadesh slots with a cognate match. This is the natural single-number
    summary of pairwise lexical similarity in glottochronology.
    """
    if a.n_meanings != b.n_meanings:
        raise ValueError("lexicons must have the same number of meaning slots")
    return float(np.mean(a.cognate_ids != b.cognate_ids))


def to_binary_matrix(
    lexicons: Sequence[Lexicon],
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Flatten lexicons into a phylogenetics-style binary character matrix.

    Each output column is a ``(meaning_slot, cognate_id)`` pair, present in
    at least one input lexicon. Cell ``X[k, j] == 1`` iff language ``k``
    carries the cognate class for that (slot, id) pair.

    This is the input format consumed by tools like BEAST 2 / RevBayes /
    MrBayes for stochastic-Dollo phylogenetic inference.

    Args:
        lexicons: Sequence of lexicons of equal length.

    Returns:
        ``(X, features)`` where ``X`` has shape ``(n_languages, n_features)``
        with dtype ``int8`` and ``features[j] == (meaning_slot, cognate_id)``.
    """
    if not lexicons:
        raise ValueError("need at least one lexicon")
    M = lexicons[0].n_meanings
    if any(lex.n_meanings != M for lex in lexicons):
        raise ValueError("all lexicons must share the same number of meaning slots")

    feature_index: dict[tuple[int, int], int] = {}
    feature_list: list[tuple[int, int]] = []
    rows: list[list[int]] = []
    for lex in lexicons:
        ids: list[int] = []
        for slot in range(M):
            key = (slot, int(lex.cognate_ids[slot]))
            j = feature_index.get(key)
            if j is None:
                j = len(feature_list)
                feature_index[key] = j
                feature_list.append(key)
            ids.append(j)
        rows.append(ids)

    n = len(lexicons)
    F = len(feature_list)
    X = np.zeros((n, F), dtype=np.int8)
    for k, ids in enumerate(rows):
        X[k, ids] = 1
    return X, feature_list
