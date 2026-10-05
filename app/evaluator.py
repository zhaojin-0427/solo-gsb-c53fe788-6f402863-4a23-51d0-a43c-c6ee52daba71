"""Relational-algebra evaluator with minimal-witness why-provenance.

Supported plan nodes (each is an object with a ``type``):

  {"type": "scan", "table": "orders"}
  {"type": "filter", "input": <node>, "predicate": <predicate tree>}
  {"type": "project", "input": <node>,
   "columns": ["a", {"from": "b", "as": "c"}]}
  {"type": "join", "left": <node>, "right": <node>,
   "on": [{"left": "id", "right": "cid"}]}
  {"type": "aggregate", "input": <node>,
   "group_by": ["cust"],
   "aggregates": [{"fn": "sum", "column": "amount", "as": "total"}]}

Every intermediate relation is a list of *labeled rows*::

    {"data": (v1, v2, ...), "witnesses": set[frozenset[label]]}

Rows with identical data are merged (set semantics); their witness sets are
unioned and pruned to inclusion-minimal witnesses. A label is a
``(table, pk-tuple)`` pair identifying one submitted source row.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

from .errors import ApiError
from .values import (
    eval_predicate,
    is_number,
    strict_equal,
    tuple_sort_key,
    type_name,
)

Label = tuple[str, tuple]
Witness = frozenset[Label]

# Column descriptor: (name, declared type or None, nullable)
Col = tuple[str, str | None]


class RecomputePool:
    """Reduced candidate pool for recomputation-based deletion-set search.

    Attributes:
      full:    every source label in the target group's member witnesses;
      active_labels: labels appearing in at least one witness of an *active*
               member (a member whose deletion can change one of the target
               aggregate values). Labels in ``full - active_labels`` support
               only inert members (e.g. zero-valued join pairs) and are the
               candidates for reduction;
      reduced: labels the exact search actually enumerates. It starts equal
               to ``active_labels``; the dependency gate
               (``provenance.refine_reduced_pool``) widens it back to ``full``
               when the inert labels may participate in a minimal deletion
               set (as with a self-join, where one source label feeds several
               members whose value changes cancel out);
      group_member_count: number of input rows in the target group;
      active_member_count: number of those that can change at least one
               aggregate value of the target output.
    """

    __slots__ = ("full", "reduced", "active_labels", "group_member_count", "active_member_count")

    def __init__(
        self,
        full: set[Label],
        reduced: set[Label],
        group_member_count: int,
        active_member_count: int,
        active_labels: set[Label] | None = None,
    ):
        self.full = full
        self.reduced = reduced
        self.active_labels = active_labels if active_labels is not None else set(reduced)
        self.group_member_count = group_member_count
        self.active_member_count = active_member_count

    @property
    def dropped_inert_count(self) -> int:
        return len(self.full) - len(self.reduced)


# ---------------------------------------------------------------------------
# Table loading / validation
# ---------------------------------------------------------------------------

def _coerce_value(v: Any, declared: str | None, *, where: str) -> Any:
    if v is None:
        return None
    if declared == "bool":
        if not isinstance(v, bool):
            raise ApiError("TYPE_ERROR", f"{where}: expected bool, got {type_name(v)}")
    elif declared == "int":
        if not isinstance(v, int) or isinstance(v, bool):
            raise ApiError("TYPE_ERROR", f"{where}: expected int, got {type_name(v)}")
    elif declared == "float":
        if not is_number(v):
            raise ApiError("TYPE_ERROR", f"{where}: expected number, got {type_name(v)}")
        return float(v)
    elif declared == "string":
        if not isinstance(v, str):
            raise ApiError("TYPE_ERROR", f"{where}: expected string, got {type_name(v)}")
    return v


class Table:
    def __init__(self, spec: dict):
        self.name: str = spec["table"]
        self.pk_cols: list[str] = list(spec["primary_key"])
        self.columns: list[Col] = []

        declared: dict[str, str] = {}
        for c in spec.get("columns", []):
            if isinstance(c, dict):
                declared[c["name"]] = c["type"]
                self.columns.append((c["name"], c["type"]))
            else:
                declared[c] = "any"
                self.columns.append((c, None))

        rows_in: list[dict] = spec.get("rows", [])
        # Infer columns from rows when (or beyond) declared columns.
        seen = {name for name, _ in self.columns}
        for r in rows_in:
            for k in r.keys():
                if k not in seen:
                    self.columns.append((k, None))
                    seen.add(k)

        self.colnames = [name for name, _ in self.columns]
        unknown_pk = [c for c in self.pk_cols if c not in self.colnames]
        if unknown_pk:
            raise ApiError(
                "ILLEGAL_REFERENCE",
                f"Table {self.name!r}: primary key column(s) {unknown_pk} not "
                f"among declared columns",
                table=self.name,
                columns=unknown_pk,
            )

        # rows: list of (pk-tuple, full data tuple aligned to columns, label)
        self.rows: list[tuple[tuple, tuple, Label]] = []
        seen_pks: dict[tuple, Label] = {}
        for idx, r in enumerate(rows_in):
            extra = [k for k in r if k not in self.colnames]
            if extra:
                raise ApiError(
                    "MALFORMED_REQUEST",
                    f"Table {self.name!r} row {idx}: unknown column(s) {extra}",
                    table=self.name,
                    row_index=idx,
                    columns=extra,
                )
            vals = []
            for name, typ in self.columns:
                v = r.get(name)
                vals.append(_coerce_value(v, typ, where=f"table {self.name!r} row {idx}, column {name!r}"))
            pk = tuple(r.get(c) for c in self.pk_cols)
            if any(p is None for p in pk):
                raise ApiError(
                    "NULL_PRIMARY_KEY",
                    f"Table {self.name!r} row {idx}: primary key part is null",
                    table=self.name,
                    row_index=idx,
                )
            label = (self.name, pk)
            if pk in seen_pks:
                raise ApiError(
                    "DUPLICATE_PRIMARY_KEY",
                    f"Table {self.name!r}: duplicate primary key {list(pk)}",
                    table=self.name,
                    primary_key=list(pk),
                )
            seen_pks[pk] = label
            self.rows.append((pk, tuple(vals), label))

    def filtered_copy(self, drop_pks: set[tuple]) -> "Table":
        """Same schema, rows whose primary key is not in *drop_pks*."""
        clone = Table.__new__(Table)
        clone.name = self.name
        clone.pk_cols = self.pk_cols
        clone.columns = self.columns
        clone.colnames = self.colnames
        clone.rows = [r for r in self.rows if r[0] not in drop_pks]
        return clone


# ---------------------------------------------------------------------------
# Witness utilities
# ---------------------------------------------------------------------------

def prune(witnesses: Iterable[Witness]) -> set[Witness]:
    """Keep only inclusion-minimal witnesses."""
    uniq = set(witnesses)
    return {w for w in uniq if not any(o < w for o in uniq)}


def cover_witnesses(rows: Iterable[dict]) -> set[Witness]:
    """Witness family of a *conjunction* of member rows.

    Every row must survive, so each witness picks one witness per row and
    unions them (cartesian product of the rows' witness families).
    """
    result: set[Witness] = {frozenset()}
    for m in rows:
        result = {a | b for a in result for b in m["witnesses"]}
    return prune(result)


def minimal_sum_combos(
    items: list[tuple[dict, Any]], target: Any
) -> list[tuple[dict, ...]]:
    """All inclusion-minimal NON-EMPTY subsets of *items* summing to *target*.

    Rows are matched by identity, so equal-valued rows yield distinct
    combinations. The empty selection is never a combination: a sum over zero
    source rows is not produced by any source row. This matters when the
    target is 0 — with mixed positive/negative values the empty set would
    otherwise "witness" the total and collapse the sum component to the
    trivial family, making the provenance (and the deletion sets derived
    from it) wrong. Once a partial selection already sums to the target no
    superset of it can be minimal, and with all values non-negative a
    partial sum above the target can never come back down; both observations
    prune the backtracking.
    """
    n = len(items)
    non_negative = all(is_number(v) and v >= 0 for _, v in items)
    combos: list[tuple[dict, ...]] = []
    # Explicit-stack backtracking over include/exclude decisions; each state
    # is (next index, accumulated sum, chosen indices).
    stack: list[tuple[int, Any, tuple[int, ...]]] = [(0, 0, ())]
    while stack:
        i, acc, chosen = stack.pop()
        if acc == target and chosen:
            combos.append(tuple(items[j][0] for j in chosen))
            continue
        if i == n or (non_negative and acc > target):
            continue
        stack.append((i + 1, acc, chosen))
        stack.append((i + 1, acc + items[i][1], chosen + (i,)))
    return combos


def merge_relation(rows: Iterable[dict]) -> list[dict]:
    """Merge labeled rows with identical data tuples; prune witnesses."""
    by_data: dict[tuple, set[Witness]] = defaultdict(set)
    for r in rows:
        by_data[r["data"]].update(r["witnesses"])
    out = [
        {"data": data, "witnesses": prune(ws)}
        for data, ws in by_data.items()
    ]
    out.sort(key=lambda r: tuple_sort_key(r["data"]))
    return out


class Evaluator:
    def __init__(self, tables: dict[str, Table]):
        self.tables = tables
        self._cache: dict[int, list[dict]] = {}

    # -- node helpers -------------------------------------------------------

    def eval(self, node: Any) -> list[dict]:
        if not isinstance(node, dict):
            raise ApiError("MALFORMED_PLAN", "Plan node must be an object")
        nid = id(node)
        if nid in self._cache:
            return self._cache[nid]
        t = node.get("type")
        if t == "scan":
            res = self._scan(node)
        elif t == "filter":
            res = self._filter(node)
        elif t == "project":
            res = self._project(node)
        elif t == "join":
            res = self._join(node)
        elif t == "aggregate":
            res = self._aggregate(node)
        else:
            raise ApiError(
                "MALFORMED_PLAN",
                f"Unknown plan node type {t!r}; expected one of "
                "scan/filter/project/join/aggregate",
            )
        self._cache[nid] = res
        return res

    def schema(self, node: Any) -> list[str]:
        t = node.get("type")
        if t == "scan":
            table = self._require_table(node)
            return table.colnames
        if t == "filter":
            self._filter_schema(node)
            return self.schema(self._input(node))
        if t == "project":
            return [asname for _, asname in self._proj_cols(node)]
        if t == "join":
            return self._join_schema(node)[0]
        if t == "aggregate":
            return self._agg_schema(node)
        raise ApiError("MALFORMED_PLAN", f"Unknown plan node type {t!r}")

    def _input(self, node: dict) -> Any:
        if "input" not in node:
            raise ApiError("MALFORMED_PLAN", f"{node.get('type')!r} node needs 'input'")
        return node["input"]

    def _filter_schema(self, node: dict) -> None:
        """Validate predicate shape and column references against input schema."""
        pred = node.get("predicate")
        if not isinstance(pred, dict):
            raise ApiError("MALFORMED_PLAN", "filter node needs a 'predicate' object")
        names = self.schema(self._input(node))

        def check(p: Any) -> None:
            if not isinstance(p, dict) or not p:
                raise ApiError(
                    "MALFORMED_PLAN",
                    "Predicate must be a non-empty object",
                )
            compound = [k for k in p if k in ("and", "or", "not")]
            if compound:
                if len(p) != 1:
                    raise ApiError(
                        "MALFORMED_PLAN",
                        f"Compound predicate {compound[0]!r} must have no sibling keys",
                    )
                key, body = compound[0], p[compound[0]]
            elif "column" in p:
                key, body = "column", p
            else:
                raise ApiError(
                    "MALFORMED_PLAN",
                    "Predicate must have key 'and'/'or'/'not'/'column'",
                )
            if key == "and" or key == "or":
                if not isinstance(body, list) or not body:
                    raise ApiError(
                        "MALFORMED_PLAN", f"{key!r} predicate needs a non-empty list"
                    )
                for sub in body:
                    check(sub)
            elif key == "not":
                check(body)
            elif key == "column":
                if not isinstance(body, dict):
                    raise ApiError("MALFORMED_PLAN", "column predicate must be object")
                col = body.get("column")
                op = body.get("op")
                if not isinstance(col, str) or not isinstance(op, str):
                    raise ApiError(
                        "MALFORMED_PLAN", "Predicate needs string 'column' and 'op'"
                    )
                if col not in names:
                    raise ApiError(
                        "ILLEGAL_REFERENCE",
                        f"Predicate references unknown column {col!r}",
                        column=col,
                        available=names,
                    )
                if op in {"is_null", "is_not_null"}:
                    return
                if op not in {"=", "<>", "<", "<=", ">", ">="}:
                    raise ApiError("MALFORMED_PLAN", f"Unknown predicate operator {op!r}")
                if "value" not in body:
                    raise ApiError("MALFORMED_PLAN", f"Predicate {op!r} needs 'value'")
                lit = body["value"]
                if not isinstance(lit, (str, int, float, bool)) and lit is not None:
                    raise ApiError(
                        "MALFORMED_PLAN",
                        "Predicate literal must be a JSON scalar or null",
                    )
            else:
                raise ApiError("MALFORMED_PLAN", f"Unknown predicate node {key!r}")

        check(pred)

    def _require_table(self, node: dict) -> Table:
        name = node.get("table")
        if not isinstance(name, str):
            raise ApiError("MALFORMED_PLAN", "scan node needs string 'table'")
        table = self.tables.get(name)
        if table is None:
            raise ApiError(
                "UNKNOWN_TABLE",
                f"Plan references table {name!r} which was not submitted",
                table=name,
            )
        return table

    def _col_index(self, node: Any, col: str) -> tuple[int, list[str]]:
        names = self.schema(node)
        if col not in names:
            raise ApiError(
                "ILLEGAL_REFERENCE",
                f"Column {col!r} not found in node output columns {names}",
                column=col,
                available=names,
            )
        return names.index(col), names

    # -- operators ----------------------------------------------------------

    def _scan(self, node: dict) -> list[dict]:
        table = self._require_table(node)
        return [
            {"data": data, "witnesses": {frozenset({label})}}
            for _, data, label in table.rows
        ]

    def _filter(self, node: dict) -> list[dict]:
        src = self.eval(self._input(node))
        pred = node.get("predicate")
        if not isinstance(pred, dict):
            raise ApiError("MALFORMED_PLAN", "filter node needs a 'predicate' object")
        names = self.schema(self._input(node))
        out = []
        for r in src:
            row = dict(zip(names, r["data"]))
            keep = eval_predicate(pred, row)
            if keep is True:
                out.append(r)
        # filter cannot introduce witness redundancy, but keep ordering stable
        out.sort(key=lambda r: tuple_sort_key(r["data"]))
        return out

    def _proj_cols(self, node: dict) -> list:
        cols = node.get("columns")
        if not isinstance(cols, list) or not cols:
            raise ApiError(
                "MALFORMED_PLAN", "project node needs a non-empty 'columns' list"
            )
        norm = []
        for c in cols:
            if isinstance(c, str):
                norm.append((c, c))
            elif isinstance(c, dict) and isinstance(c.get("from"), str) and isinstance(c.get("as"), str):
                norm.append((c["from"], c["as"]))
            else:
                raise ApiError(
                    "MALFORMED_PLAN",
                    "project columns must be strings or {from, as} objects",
                )
        return norm

    def _proj_as(self, c: Any) -> str:
        return c if isinstance(c, str) else c["as"]

    def _project(self, node: dict) -> list[dict]:
        src_node = self._input(node)
        src = self.eval(src_node)
        names = self.schema(src_node)
        spec = self._proj_cols(node)
        idx: list[int] = []
        out_names: list[str] = []
        for frm, asname in spec:
            if frm not in names:
                raise ApiError(
                    "ILLEGAL_REFERENCE",
                    f"project references unknown column {frm!r}",
                    column=frm,
                    available=names,
                )
            if asname in out_names:
                raise ApiError(
                    "MALFORMED_PLAN", f"project output has duplicate column {asname!r}"
                )
            idx.append(names.index(frm))
            out_names.append(asname)
        produced = []
        for r in src:
            produced.append(
                {
                    "data": tuple(r["data"][i] for i in idx),
                    "witnesses": set(r["witnesses"]),
                }
            )
        return merge_relation(produced)

    def _join_aliases(self, node: dict) -> tuple[str | None, str | None]:
        left_alias = node.get("left_alias")
        right_alias = node.get("right_alias")
        for a in (left_alias, right_alias):
            if a is not None and not isinstance(a, str):
                raise ApiError("MALFORMED_PLAN", "join alias must be a string")
        return left_alias, right_alias

    def _join_schema(self, node: dict) -> tuple[list[str], list[str], list[str], list[tuple[str, str]]]:
        left_node = node.get("left")
        right_node = node.get("right")
        if left_node is None or right_node is None:
            raise ApiError("MALFORMED_PLAN", "join node needs 'left' and 'right'")
        ln = self.schema(left_node)
        rn = self.schema(right_node)
        la, ra = self._join_aliases(node)
        on = self._join_on(node, ln, rn)

        out: list[str] = list(ln)
        # Equal columns on the join keys appear once (USING semantics).
        right_key_cols = {rc for _, rc in on}
        for c in rn:
            if c in right_key_cols and c in ln:
                continue
            name = c
            if name in ln or (la and name.startswith(f"{la}.")) or (ra and name.startswith(f"{ra}.")):
                q = ra or "right"
                name = f"{q}.{c}"
                if name in ln or name in out:
                    raise ApiError(
                        "MALFORMED_PLAN",
                        f"join output column name collision on {name!r}; use "
                        "right_alias or project to disambiguate",
                    )
            out.append(name)
        return out, ln, rn, on

    def _join_on(self, node: dict, ln: list[str], rn: list[str]) -> list[tuple[str, str]]:
        on = node.get("on")
        if not isinstance(on, list) or not on:
            raise ApiError("MALFORMED_PLAN", "join node needs a non-empty 'on' list")
        pairs: list[tuple[str, str]] = []
        for p in on:
            if not isinstance(p, dict) or "left" not in p or "right" not in p:
                raise ApiError(
                    "MALFORMED_PLAN", "join 'on' entries must be {left, right} objects"
                )
            lc, rc = p["left"], p["right"]
            if lc not in ln:
                raise ApiError(
                    "ILLEGAL_REFERENCE",
                    f"join references unknown left column {lc!r}",
                    column=lc,
                    available=ln,
                )
            if rc not in rn:
                raise ApiError(
                    "ILLEGAL_REFERENCE",
                    f"join references unknown right column {rc!r}",
                    column=rc,
                    available=rn,
                )
            pairs.append((lc, rc))
        return pairs

    def _join(self, node: dict) -> list[dict]:
        out_names, ln, rn, on = self._join_schema(node)
        left = self.eval(node["left"])
        right = self.eval(node["right"])
        li = [ln.index(lc) for lc, _ in on]
        ri = [rn.index(rc) for _, rc in on]

        # Hash right side on the composite equi-key. Keys containing NULL never
        # match (SQL NULL semantics); rows of different hash values simply fail
        # to join. A TYPE_ERROR is only raised for colliding keys whose values
        # are equal by Python hash/eq but not strict JSON equality (e.g. bool
        # vs int), since that is an ambiguous, almost certainly unintended
        # cross-type comparison.
        index: dict[tuple, list[dict]] = defaultdict(list)
        for r in right:
            key = tuple(r["data"][i] for i in ri)
            if any(k is None for k in key):
                continue
            index[key].append(r)

        produced = []
        for lr in left:
            lkey = tuple(lr["data"][i] for i in li)
            if any(k is None for k in lkey):
                continue
            for rr in index.get(lkey, ()):
                # strict type check across the key pairs
                for a, b in zip(lkey, (rr["data"][i] for i in ri)):
                    if strict_equal(a, b) is False:
                        raise ApiError(
                            "TYPE_ERROR",
                            "Equi-join key values have mismatched types "
                            f"({type_name(a)} vs {type_name(b)})",
                        )
                data = list(lr["data"])
                right_key_cols = {rc for _, rc in on}
                for j, c in enumerate(rn):
                    if c in right_key_cols and c in ln:
                        continue
                    data.append(rr["data"][j])
                witnesses: set[Witness] = set()
                for lw in lr["witnesses"]:
                    for rw in rr["witnesses"]:
                        witnesses.add(frozenset(lw | rw))
                produced.append({"data": tuple(data), "witnesses": witnesses})
        return merge_relation(produced)

    # -- aggregate ----------------------------------------------------------

    _AGG_FNS = {"count", "sum", "min", "max"}

    # Sentinel for the project walk in mixed_sign_sum_pool: marks input
    # columns the projection did not carry through.
    _UNMAPPED = object()

    def mixed_sign_sum_pool(self, node: Any, target_data: tuple) -> "RecomputePool | None":
        """Reduced candidate pool for the target row when its group has a
        ``sum`` over mixed positive/negative values (a non-monotone sum).

        *node* is the plan root and *target_data* the target row aligned to
        its output schema. The walk descends through ``filter``/``project``
        nodes (which neither change row identity nor dependencies) to the
        innermost ``aggregate`` producing the target. Returns ``None`` when:

        * no ``sum`` aggregate is involved (count/min/max are exact through
          hitting sets), or
        * the target group's sum values are all non-negative or all
          non-positive (the sum then changes iff a minimal-combination row is
          deleted, so hitting sets stay exact), or
        * the aggregate cannot be reached by the walk (e.g. a join above
          it), in which case the caller keeps the hitting-set answer.

        Candidate reduction
        -------------------
        A source row that cannot change *any* aggregate value of the target
        output can never belong to an inclusion-minimal deletion set for that
        full output row, so its exclusive labels are excluded from the
        returned pool. Per aggregate over the target group, an input member
        is *inert* when:

        * ``count(*)``: never inert (each member changes the count);
        * ``count(col)``: its ``col`` value is NULL (NULLs are not counted);
        * ``sum(col)``: its ``col`` value is NULL or zero (zero leaves the
          total unchanged; this removes arbitrary many padding zero rows);
        * ``min``/``max(col)``: its non-NULL value differs from the target
          extremum — removing it cannot move the extremum.

        A member is *active* when it is non-inert for at least one aggregate.
        The provisional reduced pool is the union of the active members'
        witness labels (any label supporting only inert members is dropped
        for now). That provisional reduction is exact when every group member
        depends on its own source labels — scans, filters, and one-to-one
        joins: deleting one active label then removes exactly one active
        member, so every active-label singleton already changes the target.
        It is NOT exact in general under joins, where one source label can
        feed several members: deleting it may remove a +5 member and a -5
        member together (net zero) while a zero-valued self-join member
        survives, leaving the target in place — the inert member's labels
        then belong to a minimal deletion set after all. The caller
        therefore validates the reduction with the label-dependency gate in
        ``provenance.refine_reduced_pool`` (recomputing once per active
        label) and widens the pool back to ``full`` when the gate fails.
        """
        data = tuple(target_data)
        while True:
            t = node.get("type") if isinstance(node, dict) else None
            if t == "filter":
                node = self._input(node)
            elif t == "project":
                spec = self._proj_cols(node)
                out_values = dict(zip([asname for _, asname in spec], data))
                in_names = self.schema(self._input(node))
                mapped = {name: self._UNMAPPED for name in in_names}
                for frm, asname in spec:
                    mapped[frm] = out_values[asname]
                data = tuple(mapped[name] for name in in_names)
                node = self._input(node)
            elif t == "aggregate":
                break
            else:
                return None

        gb = node.get("group_by", [])
        aggs = self._agg_spec(node)
        sum_cols = [col for fn, col, _ in aggs if fn == "sum"]
        if not sum_cols:
            return None
        out_schema = self._agg_schema(node)
        values = dict(zip(out_schema, data))
        key = tuple(values[g] for g in gb)
        if any(v is self._UNMAPPED for v in key):
            return None
        in_names = self.schema(self._input(node))
        gi = [in_names.index(g) for g in gb]
        members = [
            r
            for r in self.eval(self._input(node))
            if tuple(r["data"][i] for i in gi) == key
        ]
        mixed = False
        for col in sum_cols:
            ci = in_names.index(col)
            vals = [r["data"][ci] for r in members if r["data"][ci] is not None]
            if any(v > 0 for v in vals) and any(v < 0 for v in vals):
                mixed = True
                break
        if not mixed:
            return None

        # Target aggregate values, aligned to the spec: all-NULL sums/mins
        # are None. A member is active iff deleting it can change at least
        # one target aggregate value (see the class docstring).
        active = [False] * len(members)
        for fn, col, asname in aggs:
            target = values[asname]
            if fn == "count":
                if col is None or col == "*":
                    for i in range(len(members)):
                        active[i] = True
                else:
                    ci = in_names.index(col)
                    for i, m in enumerate(members):
                        if m["data"][ci] is not None:
                            active[i] = True
            elif fn == "sum":
                ci = in_names.index(col)
                for i, m in enumerate(members):
                    v = m["data"][ci]
                    if v is not None and v != 0:
                        active[i] = True
            else:  # min / max
                ci = in_names.index(col)
                for i, m in enumerate(members):
                    v = m["data"][ci]
                    if v is not None and target is not None and v == target:
                        active[i] = True

        full_pool: set[Label] = set()
        reduced_pool: set[Label] = set()
        for i, r in enumerate(members):
            member_labels: set[Label] = set()
            for w in r["witnesses"]:
                member_labels.update(w)
            full_pool.update(member_labels)
            if active[i]:
                reduced_pool.update(member_labels)

        # Defensive: an empty active pool would mean no active label at all,
        # but a mixed-sign sum guarantees non-zero members exist; leave the
        # exact-path decision to the caller rather than searching nothing.
        if not reduced_pool:
            return None

        # The reduced pool initially holds only the active members' labels.
        # Whether labels supporting inert members alone (e.g. a zero-valued
        # self-join pair) may be dropped is decided afterwards by the label
        # dependency gate in provenance.refine_reduced_pool: with joins one
        # source label can feed several members whose value changes cancel
        # (e.g. the +5 and -5 pairs sharing source row 1), so an inert member
        # can be exactly what keeps the zero target alive after its active
        # partners are removed.
        return RecomputePool(
            full=full_pool,
            reduced=reduced_pool,
            group_member_count=len(members),
            active_member_count=sum(active),
            active_labels=set(reduced_pool),
        )

    def _agg_spec(self, node: dict):
        aggs = node.get("aggregates", [])
        if not isinstance(aggs, list) or not aggs:
            raise ApiError(
                "MALFORMED_PLAN", "aggregate node needs a non-empty 'aggregates' list"
            )
        norm = []
        for a in aggs:
            if not isinstance(a, dict) or not isinstance(a.get("fn"), str):
                raise ApiError("MALFORMED_PLAN", "aggregate entries need string 'fn'")
            fn = a["fn"]
            if fn not in self._AGG_FNS:
                raise ApiError(
                    "MALFORMED_PLAN",
                    f"Unsupported aggregate {fn!r}; allowed: count/sum/min/max",
                )
            col = a.get("column")
            if fn != "count" and not isinstance(col, str):
                raise ApiError(
                    "MALFORMED_PLAN", f"{fn} aggregate requires a string 'column'"
                )
            if fn == "count" and col is not None and not isinstance(col, str):
                raise ApiError(
                    "MALFORMED_PLAN", "count column must be a string or '*'"
                )
            asname = a.get("as")
            if not isinstance(asname, str) or not asname:
                raise ApiError("MALFORMED_PLAN", "aggregate requires string 'as'")
            norm.append((fn, col, asname))
        names = [a[2] for a in norm]
        if len(set(names)) != len(names):
            raise ApiError("MALFORMED_PLAN", "aggregate output aliases must be unique")
        return norm

    def _agg_schema(self, node: dict) -> list[str]:
        gb = node.get("group_by", [])
        if not isinstance(gb, list) or not all(isinstance(c, str) for c in gb):
            raise ApiError("MALFORMED_PLAN", "aggregate 'group_by' must be strings")
        out = list(gb) + [a[2] for a in self._agg_spec(node)]
        if len(set(out)) != len(out):
            raise ApiError(
                "MALFORMED_PLAN", "aggregate output column names must be unique"
            )
        return out

    def _agg_component(
        self, fn: str, col: str | None, names: list[str], members: list[dict]
    ) -> tuple[Any, set[Witness]]:
        """Value and witness family of one aggregate expression over a group.

        The family characterises the member-row sets that still produce the
        value, so that the minimal deletion sets (hitting sets of the row's
        witnesses) are exactly replayed by recomputation:

        * ``count(*)`` / ``count(col)`` covers every row participating in the
          count (all members, resp. all members with a non-NULL value): one
          conjunction over those rows;
        * ``sum(col)`` enumerates the inclusion-minimal NON-EMPTY row
          combinations whose values add up to the target sum;
        * ``min(col)`` / ``max(col)`` enumerates the inclusion-minimal
          combinations still yielding the extremum — one per member row equal
          to it (equal-valued rows give distinct combinations).

        NULLs never participate in ``count(col)``/``sum``/``min``/``max``; an
        all-NULL aggregate (value NULL) constrains nothing beyond group
        existence and contributes the trivial family ``{∅}``.
        """
        if fn == "count":
            if col is None or col == "*":
                counted = list(members)
            else:
                ci = names.index(col)
                counted = [m for m in members if m["data"][ci] is not None]
            return len(counted), cover_witnesses(counted)

        ci = names.index(col)
        vals = [(m, m["data"][ci]) for m in members if m["data"][ci] is not None]
        if fn == "sum":
            bad = [v for _, v in vals if not is_number(v)]
            if bad:
                raise ApiError(
                    "TYPE_ERROR",
                    f"sum({col!r}): non-numeric value "
                    f"{bad[0]!r} ({type_name(bad[0])})",
                    column=col,
                )
            if not vals:
                return None, {frozenset()}
            target = 0
            for _, v in vals:
                target += v
            family: set[Witness] = set()
            for combo in minimal_sum_combos(vals, target):
                family |= cover_witnesses(combo)
            return target, prune(family)

        # min / max
        kinds = {type_name(v) for _, v in vals}
        non_numeric = kinds - {"number"}
        if non_numeric and ({"number"} & kinds or len(non_numeric) > 1):
            raise ApiError(
                "TYPE_ERROR",
                f"{fn}({col!r}): cannot order mixed types {sorted(kinds)}",
                column=col,
            )
        if not vals:
            return None, {frozenset()}
        target = (min if fn == "min" else max)(v for _, v in vals)
        family = set()
        for m, v in vals:
            if v == target:
                family |= m["witnesses"]
        return target, prune(family)

    def _aggregate(self, node: dict) -> list[dict]:
        src_node = self._input(node)
        gb = node.get("group_by", [])
        aggs = self._agg_spec(node)
        names = self.schema(src_node)

        for c in gb:
            if c not in names:
                raise ApiError(
                    "ILLEGAL_REFERENCE",
                    f"group_by references unknown column {c!r}",
                    column=c,
                    available=names,
                )
        for fn, col, asname in aggs:
            if fn != "count" and col not in names:
                raise ApiError(
                    "ILLEGAL_REFERENCE",
                    f"aggregate {asname!r} references unknown column {col!r}",
                    column=col,
                    available=names,
                )
            if fn == "count" and col not in (None, "*") and col not in names:
                raise ApiError(
                    "ILLEGAL_REFERENCE",
                    f"aggregate {asname!r} references unknown column {col!r}",
                    column=col,
                    available=names,
                )

        src = self.eval(src_node)

        if not src:
            # Aggregating an empty relation is always rejected: for grouped
            # aggregates it would silently yield zero groups, for scalar
            # aggregates a single row of NULL/0 — the client must distinguish
            # "no input" explicitly.
            raise ApiError(
                "EMPTY_AGGREGATE",
                "aggregate input is empty: no groups can be formed",
            )

        gi = [names.index(c) for c in gb]

        # NULL group keys form their own group. We key the dict with a
        # sentinel-prefixed tuple so a leading literal value can never collide
        # with the marker, and restore the real key when emitting.
        NULL_KEY = object()
        groups: dict[tuple, list[dict]] = defaultdict(list)
        for r in src:
            gkey = tuple(r["data"][i] for i in gi)
            if any(k is None for k in gkey):
                groups[(NULL_KEY,) + gkey].append(r)
            else:
                groups[gkey].append(r)

        out_rows = []
        for gkey, members in groups.items():
            real_key = gkey[1:] if gkey and gkey[0] is NULL_KEY else gkey
            data: list[Any] = list(real_key)
            # Witness components that must all hold for this exact output row:
            #   1. group existence — at least one member row survives;
            #   2. one component per aggregate expression pinning its value.
            # The row's witnesses are the conjunction (pairwise union) of the
            # component families, pruned to inclusion-minimal sets.
            families: list[set[Witness]] = []
            existence: set[Witness] = set()
            for m in members:
                existence.update(m["witnesses"])
            families.append(prune(existence))
            for fn, col, asname in aggs:
                value, family = self._agg_component(fn, col, names, members)
                data.append(value)
                families.append(family)
            witnesses: set[Witness] = {frozenset()}
            for family in families:
                witnesses = prune({a | b for a in witnesses for b in family})
            out_rows.append({"data": tuple(data), "witnesses": witnesses})
        return merge_relation(out_rows)
