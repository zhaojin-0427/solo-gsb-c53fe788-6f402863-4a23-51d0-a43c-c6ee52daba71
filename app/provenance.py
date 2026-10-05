"""Provenance presentation and counterfactual (minimal deletion sets).

Boolean provenance
------------------
Each submitted source row receives a short symbol ``s1, s2, ...``. A minimal
witness set is a conjunction of symbols; the provenance of an output row is
the DNF over all its minimal witnesses::

    (s1 ∧ s3) ∨ (s2 ∧ s3)

Because witnesses are inclusion-minimal and the operators are monotone in
the source rows (select / equi-join / project; aggregates pin their exact
output value — count covers every counted row, sum/min/max enumerate the
minimal non-empty row combinations producing the target value), the sets of
source rows whose removal makes an output row disappear are exactly the
minimal hitting sets of its minimal witnesses — EXCEPT for ``sum`` over
mixed positive/negative values. Such a sum is non-monotone: deleting a row
that belongs to no minimal combination can still change the total, and
deleting further rows may even bring the target value back. No witness
family characterises those deletion sets, so they are determined by
recomputation instead: see ``minimal_deletion_sets_by_recompute``.
"""
from __future__ import annotations

import itertools
from typing import Any, Callable, Iterable

from .evaluator import Label, Witness
from .values import tuple_sort_key, value_sort_key


# ---------------------------------------------------------------------------
# Labels / symbols
# ---------------------------------------------------------------------------

def label_to_string(label: Label) -> str:
    table, pk = label
    parts = [_pk_part(p) for p in pk]
    return f"{table}[" + ",".join(parts) + "]"


def _pk_part(p: Any) -> str:
    if isinstance(p, str):
        return p
    if p is None:
        return "null"
    if isinstance(p, bool):
        return "true" if p else "false"
    return str(p)


def label_sort_key(label: Label):
    return (label[0], tuple(value_sort_key(p) for p in label[1]))


def build_symbol_table(all_labels: set[Label]) -> dict[Label, str]:
    ordered = sorted(all_labels, key=label_sort_key)
    return {label: f"s{i + 1}" for i, label in enumerate(ordered)}


# ---------------------------------------------------------------------------
# Row presentation
# ---------------------------------------------------------------------------

def present_row(
    row: dict,
    columns: list[str],
    symbols: dict[Label, str],
) -> dict:
    witnesses = sorted(
        sorted(w, key=lambda lab: label_sort_key(lab)) for w in row["witnesses"]
    )
    witness_objs = [
        {
            "labels": [label_to_string(lab) for lab in w],
            "symbols": [symbols[lab] for lab in w],
            "conjunction": " ∧ ".join(symbols[lab] for lab in w),
        }
        for w in witnesses
    ]
    return {
        "row": {c: v for c, v in zip(columns, row["data"])},
        "witness_count": len(witness_objs),
        "minimal_witnesses": witness_objs,
        "provenance": {
            "dnf": " ∨ ".join(f"({w['conjunction']})" for w in witness_objs),
            "dnf_structured": [w["symbols"] for w in witness_objs],
            "labels_structured": [w["labels"] for w in witness_objs],
        },
    }


# ---------------------------------------------------------------------------
# Minimal hitting sets (minimal transversals)
# ---------------------------------------------------------------------------

def deletion_set_sort_key(h: Iterable[Label]):
    """Stable total order for deletion sets, by source label."""
    return [label_sort_key(l) for l in sorted(h, key=label_sort_key)]


def minimal_hitting_sets(witnesses: set[Witness]) -> list[frozenset[Label]]:
    """All inclusion-minimal sets H with H ∩ W ≠ ∅ for every W."""
    edges = [frozenset(w) for w in witnesses if w]
    memo: dict[frozenset, set[frozenset[Label]]] = {}

    def rec(edges: tuple[frozenset, ...]) -> set[frozenset[Label]]:
        key = frozenset(edges)
        if key in memo:
            return memo[key]
        if not edges:
            return {frozenset()}

        # Pick the smallest edge to branch on (keeps the recursion narrow).
        e = min(edges, key=len)
        results: set[frozenset[Label]] = set()
        for v in e:
            # Choosing v covers e and every edge containing v; the residual
            # hypergraph is all remaining edges (intact) that v does not hit.
            remaining = tuple(x for x in edges if v not in x)
            for sub in rec(remaining):
                results.add(frozenset(sub | {v}))
        # Prune non-minimal solutions introduced on this branch.
        minimal = {h for h in results if not any(o < h for o in results)}
        memo[key] = minimal
        return minimal

    hitting = rec(tuple(edges))
    return sorted(hitting, key=deletion_set_sort_key)


# ---------------------------------------------------------------------------
# Recomputation-based minimal deletion sets (non-monotone sum)
# ---------------------------------------------------------------------------

# After candidate reduction (see Evaluator.mixed_sign_sum_pool) the pool only
# contains source rows that can change at least one aggregate value of the
# target group; even so, exhaustive subset enumeration is 2**n, so the
# reduced pool is capped. Beyond the cap the request fails with a structured
# budget error — a hitting-set approximation is never returned as success.
MAX_RECOMPUTE_POOL = 20

# Cap on whole-plan recomputations per counterfactual search. Every candidate
# deletion set is validated by recomputing the full plan on the filtered
# snapshot; this bounds the worst-case cost of one /query call.
MAX_RECOMPUTE_CALLS = 10_000


class SearchBudgetExceeded(Exception):
    """Exact deletion-set enumeration could not finish within budget.

    ``reason`` is ``"pool_too_large"`` (reduced candidate pool over the
    enumeration cap) or ``"recompute_budget"`` (call cap hit mid-search).
    ``details`` carries machine-readable statistics merged with pool
    statistics by the caller into a structured error response.
    """

    def __init__(self, reason: str, **details: object):
        super().__init__(reason)
        self.reason = reason
        self.details = details


def refine_reduced_pool(
    pool_labels: Iterable[Label],
    active_labels: Iterable[Label],
    full_labels: Iterable[Label],
    kills: Callable[[frozenset[Label]], bool],
    *,
    max_calls: int = MAX_RECOMPUTE_CALLS,
) -> tuple[set[Label], dict[frozenset[Label], bool], int]:
    """Label-dependency gate for inert-label candidate reduction.

    ``pool_labels`` is the provisional reduced pool (labels of active
    members); ``full_labels`` adds labels that support only inert members
    (e.g. a zero-valued join pair). Dropping the inert-only labels is proven
    safe exactly when deleting **each** active label on its own already
    removes the full target row:

    * inert-only deletions never remove an active member (their labels occur
      in no active member's witnesses) and an inert member changes no
      aggregate value, so an inert-only deletion cannot kill the target;
    * if every active-label singleton kills, any deletion set containing an
      inert label also contains an active singleton (the set must remove at
      least one active member to change the target), and is therefore not
      inclusion-minimal.

    Under scans / filters / one-to-one joins every active label owns its
    members, so each active singleton changes a mixed-sign sum and the gate
    passes — arbitrary many zero/NULL members stay reduced out. Under joins
    one label may feed several members whose changes cancel (e.g. the +5 and
    -5 pairs sharing source row 1, with a zero-valued row-4 self-join kept
    alive): its singleton does NOT kill, the gate fails, and the pool is
    widened back to ``full_labels`` so exhaustive search can find the
    two-label sets (e.g. ``{row1, row4}``).

    Returns ``(pool, outcomes, calls)``: the pool to enumerate, the singleton
    kill outcomes (reused by the search without recomputation), and the
    number of recomputations spent. Raises ``SearchBudgetExceeded`` if the
    gate itself exceeds ``max_calls`` recomputations.
    """
    pool = set(pool_labels)
    active = sorted(set(active_labels), key=label_sort_key)
    outcomes: dict[frozenset[Label], bool] = {}
    calls = 0
    for lab in active:
        singleton = frozenset({lab})
        calls += 1
        if calls > max_calls:
            raise SearchBudgetExceeded(
                "recompute_budget",
                reduced_pool_size=len(pool),
                recompute_calls=calls,
                recompute_budget_limit=max_calls,
            )
        result = kills(singleton)
        outcomes[singleton] = result
        if not result:
            # An active label whose deletion does not kill the target: some
            # minimal deletion set must combine it with further labels. Those
            # further labels may be the inert-only ones, so keep the full
            # pool. (The remaining active singletons still have to be tested
            # by the caller as ordinary candidates.)
            return set(full_labels), outcomes, calls
    return pool, outcomes, calls


def minimal_deletion_sets_by_recompute(
    pool: Iterable[Label],
    kills: Callable[[frozenset[Label]], bool],
    *,
    max_pool_size: int = MAX_RECOMPUTE_POOL,
    max_calls: int = MAX_RECOMPUTE_CALLS,
    known: dict[frozenset[Label], bool] | None = None,
) -> list[frozenset[Label]]:
    """All inclusion-minimal subsets D of *pool* for which ``kills(D)`` holds.

    ``kills(D)`` recomputes the whole plan with the rows in D removed and
    reports whether the full target row is gone (a recompute error such as
    EMPTY_AGGREGATE counts as gone). Unlike hitting sets, the solutions of a
    non-monotone sum are not upward-closed — deleting more rows can make the
    target value reappear — so candidates are enumerated in increasing size.

    Pruning rule: a candidate is skipped **only** when it is a superset of an
    already-found solution (it could never be inclusion-minimal). The fact
    that some smaller set did *not* kill the target never prunes its
    supersets — in a mixed-sign sum deleting further rows may well kill a
    target the smaller deletion left alive.

    Raises ``SearchBudgetExceeded`` when the reduced pool exceeds
    ``max_pool_size`` or when more than ``max_calls`` recomputations would be
    needed; no partial/approximate answer is returned. ``known`` carries
    candidate outcomes already computed (e.g. the singleton outcomes from
    ``refine_reduced_pool``) so they are not recomputed; those prior
    recomputations are not part of ``max_calls`` — the caller accounts for
    them against the shared budget.
    """
    labels = sorted(set(pool), key=label_sort_key)
    if len(labels) > max_pool_size:
        raise SearchBudgetExceeded(
            "pool_too_large",
            reduced_pool_size=len(labels),
            pool_limit=max_pool_size,
        )

    known = known or {}
    solutions: list[frozenset[Label]] = []
    calls = 0

    def evaluate(candidate: frozenset[Label]) -> bool:
        nonlocal calls
        cached = known.get(candidate)
        if cached is not None or candidate in known:
            return cached
        calls += 1
        if calls > max_calls:
            raise SearchBudgetExceeded(
                "recompute_budget",
                reduced_pool_size=len(labels),
                recompute_calls=calls,
                recompute_budget_limit=max_calls,
            )
        return kills(candidate)

    for size in range(len(labels) + 1):
        for combo in itertools.combinations(labels, size):
            candidate = frozenset(combo)
            if any(s <= candidate for s in solutions):
                continue
            if evaluate(candidate):
                solutions.append(candidate)
    return sorted(solutions, key=deletion_set_sort_key)
