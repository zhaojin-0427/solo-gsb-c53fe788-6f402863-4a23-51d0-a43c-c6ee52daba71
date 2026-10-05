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
    pool: "RecomputePool",
    sustains: Callable[[frozenset[Label]], bool],
    *,
    max_calls: int = MAX_RECOMPUTE_CALLS,
) -> tuple[set[Label], dict[frozenset[Label], bool], int]:
    """Structural label-dependency gate for inert-label candidate reduction.

    ``pool`` is the provisional reduction of
    ``Evaluator.mixed_sign_sum_pool``: its reduced pool is the active label
    set ``A``; ``I = full - A`` supports only inert members (zero/NULL sum
    members, non-extremal min/max members, NULL ``count(col)`` members).

    The old gate recomputed one whole plan per active singleton and, on the
    first non-killing singleton, widened the pool back to *every* member
    label. Under a self-join that is necessary (label 1 feeds both the +5 and
    -5 members; its singleton removal nets zero while a zero self-join member
    survives), but it is not sufficient: arbitrary many independent zero
    members then inflate the pool past its cap and produce a false
    ``COUNTERFACTUAL_SEARCH_BUDGET_EXCEEDED`` even though none of their labels
    can belong to a minimal deletion set (a non-zero target, e.g. total=7,
    cannot be sustained by zero-valued survivors).

    Instead the gate decides structurally, simulating label deletions
    directly on the target group's labeled members via *sustains*:

    1. Every active-label singleton is evaluated; if each kills, ``I`` stays
       reduced out (scans / filters / one-to-one joins).
    2. Otherwise an inert-only label ``z`` is admitted exactly when it can
       belong to an inclusion-minimal killing set: there is an inert-member
       witness ``e ∋ z``, an active side ``X ⊆ A \\ e`` hitting every active
       member's witnesses, and an inert side ``H0 ⊆ I \\ (e \\ {z})`` hitting
       every witness that ``X`` and ``z`` miss, such that

          * deleting ``X ∪ H0 ∪ {z}`` kills (no surviving member);
          * deleting ``X ∪ H0`` leaves the full target row in place (the
            members surviving through z-witnesses re-aggregate to it);
          * every subset of ``X`` still sustains the target with all inert
            members alive — so no smaller killing set avoiding ``z`` exists
            (this rejects the padding zero labels of a one-to-one join,
            where deleting just one of the +/- partners already kills).

    For plain sums all-inert survivors can only re-aggregate to zero/NULL, so
    padding zero labels are admitted for a zero target (they may be the only
    thing keeping the row alive) and rejected for a non-zero target. Each
    structural evaluation is one target-group replay and counts against
    ``max_calls``; the gate raises ``SearchBudgetExceeded`` when the cap is
    exhausted. Returns ``(pool, singleton_outcomes, evaluations)``; the
    singleton outcomes are exact (the local structural replay agrees with the
    whole-plan recomputation for the target row) and seed the enumeration
    cache.
    """
    active = set(pool.active_labels)
    active_sorted = sorted(active, key=label_sort_key)
    outcomes: dict[frozenset[Label], bool] = {}
    cache: dict[frozenset[Label], bool] = {}
    calls = 0

    def evaluate(removed: set[Label]) -> bool:
        nonlocal calls
        key = frozenset(removed)
        cached = cache.get(key)
        if cached is not None or key in cache:
            return cached
        calls += 1
        if calls > max_calls:
            raise SearchBudgetExceeded(
                "recompute_budget",
                reduced_pool_size=len(pool.full),
                recompute_calls=calls,
                recompute_budget_limit=max_calls,
            )
        result = sustains(key)
        cache[key] = result
        return result

    # Step 1: active-label singletons.
    failing = False
    for lab in active_sorted:
        alive = evaluate({lab})
        outcomes[frozenset({lab})] = not alive
        if alive:
            failing = True

    if not failing:
        return active, outcomes, calls

    # Step 2: admit exactly the inert-only labels that can participate in an
    # inclusion-minimal killing set.
    admitted = _admit_inert_labels(pool, active, evaluate)
    return active | admitted, outcomes, calls


def _admit_inert_labels(
    pool: "RecomputePool",
    active: set[Label],
    evaluate: Callable[[set[Label]], bool],
) -> set[Label]:
    """Inert-only labels passing the structural dependency test."""
    inert_labels = set(pool.full) - active
    if not inert_labels:
        return set()

    # Witness families of active vs inert members (a member is one aggregate
    # input row; projection may merge equal-valued members, unioning their
    # witnesses, and every such witness is kept separately here).
    active_edges: set[Witness] = set()
    inert_witness_sets: set[Witness] = set()
    for member, is_active in zip(pool.member_rows, pool.member_active):
        if is_active:
            active_edges.update(member["witnesses"])
        else:
            inert_witness_sets.update(member["witnesses"])
    if not active_edges or not inert_witness_sets:
        return set()

    admitted: set[Label] = set()
    for z in sorted(inert_labels, key=label_sort_key):
        e_options = sorted(
            (w for w in inert_witness_sets if z in w),
            key=lambda w: [label_sort_key(l) for l in sorted(w, key=label_sort_key)],
        )
        for e in e_options:
            if _inert_label_essential(
                pool, active, inert_labels, active_edges, inert_witness_sets, z, e, evaluate
            ):
                admitted.add(z)
                break
    return admitted


def _inert_label_essential(
    pool: "RecomputePool",
    active: set[Label],
    inert_labels: set[Label],
    active_edges: set[Witness],
    inert_edges: set[Witness],
    z: Label,
    e: Witness,
    evaluate: Callable[[set[Label]], bool],
) -> bool:
    """Test one witness ``e ∋ z`` for an inert-only label ``z``."""
    # X may only use active labels outside e: labels of e\{z} must stay
    # available as the witnesses through which the member survives after
    # deleting X ∪ H0.
    domain = active - e
    # An active edge with no vertex in the domain can be hit neither by X nor
    # by z/H0 (H0 uses inert-only labels, z is inert), so no construction
    # exists for this witness e.
    if any(edge.isdisjoint(domain) for edge in active_edges):
        return False

    # Restrict X to inclusion-minimal transversals of the active edges. With
    # a minimal transversal, every Y ⊊ X leaves an active member alive, so
    # deleting inert-only labels in addition to Y cannot empty the group or
    # change the aggregate values — the minimality checks below are then
    # exact. Enumerating only minimal transversals also keeps the gate cheap.
    candidates_x = minimal_hitting_sets(active_edges)
    candidates_x = [
        set(x) for x in candidates_x if x <= domain
    ]
    candidates_x.sort(key=lambda x: [label_sort_key(l) for l in sorted(x, key=label_sort_key)])
    for x_set in candidates_x:
        # H0 must hit every inert witness that X misses and that does not
        # contain z (z itself is deleted in D*, killing all z-witnesses),
        # using only inert labels outside e. Taking the union of the
        # available vertices hits all such edges at once; if one edge has no
        # available vertex, this X cannot work.
        h0: set[Label] = set()
        feasible = True
        for w in inert_edges:
            if not w.isdisjoint(x_set) or z in w:
                continue
            vertices = {v for v in w - e if v in inert_labels}
            if not vertices:
                feasible = False
                break
            h0 |= vertices
        if not feasible:
            continue

        # D* = X ∪ H0 ∪ {z} kills (construction); D*\{z} must sustain the
        # full target row (the e-member survives via e, disjoint from
        # X ∪ H0).
        if not evaluate(x_set | h0):
            continue

        # No smaller killing set avoiding z: every Y ⊊ X leaves an active
        # member alive (X is a minimal transversal), whose values — inert
        # members contribute nothing — match the target (sustains(Y));
        # deleting inert-only labels on top neither empties the group nor
        # moves the values. Y = ∅ is the original target row.
        x_labels = tuple(sorted(x_set, key=label_sort_key))
        minimal = True
        for sub_size in range(len(x_labels)):
            for sub in itertools.combinations(x_labels, sub_size):
                if not evaluate(set(sub)):
                    minimal = False
                    break
            if not minimal:
                break
        if minimal:
            return True
    return False


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
