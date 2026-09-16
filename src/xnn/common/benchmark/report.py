"""Tabulate benchmark results and write them in user-selectable formats.

A benchmark result is a *table*: a list of row dicts sharing a common set of
columns (``model`` and ``split`` first, then one column per ``target_metric``).
:func:`columns` derives the ordered column list from the rows.

Writers follow the same name -> callable registry pattern as the model and
metric registries, so new output formats -- including user-defined ones -- are
added without touching the core. The built-ins are ``csv``, ``json`` and
``md`` (GitHub-flavored Markdown). :func:`write` dispatches one format;
:func:`write_all` writes several and returns the paths. :func:`format_table`
renders the same rows as an aligned plain-text table for stdout.
"""
from __future__ import annotations

import json
import os
from typing import Any, Callable

Row = dict[str, Any]
Writer = Callable[[list[Row], list[str], str], None]

_WRITERS: dict[str, Writer] = {}


def register_writer(name: str) -> Callable[[Writer], Writer]:
    """Return a decorator registering a results writer under ``name``.

    A writer is called ``writer(rows, columns, path)`` and is responsible for
    serializing the rows (in column order) to ``path``.

    Parameters
    ----------
    name : str
        Format name / file extension (case-insensitive) the writer handles.

    Returns
    -------
    Callable[[Writer], Writer]
        A decorator that registers the callable it wraps and returns it
        unchanged.

    Raises
    ------
    KeyError
        When applied, if ``name`` is already registered to a different writer.

    Examples
    --------
    >>> @register_writer("tsv")
    ... def write_tsv(rows, columns, path):
    ...     with open(path, "w") as f:
    ...         f.write("\\t".join(columns) + "\\n")
    ...         for r in rows:
    ...             f.write("\\t".join(str(r.get(c, "")) for c in columns) + "\\n")
    """
    def deco(fn: Writer) -> Writer:
        key = name.lower()
        if key in _WRITERS and _WRITERS[key] is not fn:
            raise KeyError(f"writer '{name}' already registered")
        _WRITERS[key] = fn
        return fn
    return deco


def available_writers() -> list[str]:
    """List the names of all registered results writers.

    Returns
    -------
    list of str
        The registered format names, sorted alphabetically.
    """
    return sorted(_WRITERS)


def _unit_label(col: str, units: dict[str, str] | None) -> str:
    """Return the column's display label, suffixed with ``[unit]`` if it has one.

    Parameters
    ----------
    col : str
        The column name.
    units : dict of str to str or None
        Physical unit per column; ``col`` absent (or ``units`` empty/``None``)
        yields the plain name.

    Returns
    -------
    str
        ``"col [unit]"`` when a unit is known, otherwise ``col``.
    """
    unit = (units or {}).get(col)
    return f"{col} [{unit}]" if unit else col


def apply_units(rows: list[Row], units: dict[str, str] | None) -> list[Row]:
    """Relabel row keys with their units, for unit-annotated file output.

    Each key that has a unit becomes ``"key [unit]"`` (see :func:`_unit_label`),
    so writers emit the unit in the header / field names exactly as the printed
    table shows it. Rows without any unitful column are returned unchanged.

    Parameters
    ----------
    rows : list of dict
        Result rows with plain keys.
    units : dict of str to str or None
        Physical unit per column.

    Returns
    -------
    list of dict
        New rows with unit-annotated keys (order preserved), or ``rows``
        unchanged when ``units`` is empty.
    """
    if not units:
        return rows
    return [{_unit_label(k, units): v for k, v in r.items()} for r in rows]


def columns(rows: list[Row]) -> list[str]:
    """Derive the ordered column list spanning all result rows.

    ``model`` and ``split`` are placed first (when present); the remaining
    columns follow in first-seen order across the rows, so metric columns keep
    the order they were produced in.

    Parameters
    ----------
    rows : list of dict
        The result rows.

    Returns
    -------
    list of str
        Ordered column names covering every key that appears in any row.
    """
    lead = [c for c in ("model", "split") if any(c in r for r in rows)]
    seen = set(lead)
    rest: list[str] = []
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                rest.append(k)
    return lead + rest


@register_writer("csv")
def write_csv(rows: list[Row], cols: list[str], path: str) -> None:
    """Write rows as CSV with a header line.

    Parameters
    ----------
    rows : list of dict
        Result rows.
    cols : list of str
        Ordered column names (missing cells are written empty).
    path : str
        Destination file path.
    """
    import csv
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow({c: r.get(c, "") for c in cols})


@register_writer("json")
def write_json(rows: list[Row], cols: list[str], path: str) -> None:
    """Write rows as a JSON array of objects (column order preserved).

    Parameters
    ----------
    rows : list of dict
        Result rows.
    cols : list of str
        Ordered column names used to order each object's keys.
    path : str
        Destination file path.
    """
    ordered = [{c: r[c] for c in cols if c in r} for r in rows]
    with open(path, "w") as f:
        json.dump(ordered, f, indent=2)


@register_writer("md")
def write_markdown(rows: list[Row], cols: list[str], path: str) -> None:
    """Write rows as a GitHub-flavored Markdown table.

    Parameters
    ----------
    rows : list of dict
        Result rows.
    cols : list of str
        Ordered column names.
    path : str
        Destination file path.
    """
    with open(path, "w") as f:
        f.write("| " + " | ".join(cols) + " |\n")
        f.write("| " + " | ".join("---" for _ in cols) + " |\n")
        for r in rows:
            f.write("| " + " | ".join(_fmt(r.get(c, "")) for c in cols) + " |\n")


def _fmt(v: Any) -> str:
    """Format a cell value: floats to 6 significant figures, else ``str``.

    Parameters
    ----------
    v : Any
        The cell value.

    Returns
    -------
    str
        A compact string representation.
    """
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)


def write(rows: list[Row], fmt: str, path: str,
          cols: list[str] | None = None) -> str:
    """Write results in a single registered format.

    Parameters
    ----------
    rows : list of dict
        Result rows.
    fmt : str
        Registered writer name (case-insensitive).
    path : str
        Destination file path.
    cols : list of str or None, optional
        Column order; derived via :func:`columns` when ``None``.

    Returns
    -------
    str
        The ``path`` written to.

    Raises
    ------
    KeyError
        If ``fmt`` is not a registered writer.
    """
    key = fmt.lower()
    if key not in _WRITERS:
        raise KeyError(
            f"unknown output format '{fmt}'. registered: {available_writers()}")
    cols = columns(rows) if cols is None else cols
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    _WRITERS[key](rows, cols, path)
    return path


def write_all(rows: list[Row], formats: list[str], out_dir: str,
              filename: str) -> list[str]:
    """Write results to every requested format under one directory.

    Parameters
    ----------
    rows : list of dict
        Result rows.
    formats : list of str
        Registered writer names; each appends its own extension to
        ``filename``.
    out_dir : str
        Output directory (created if needed).
    filename : str
        Base filename without extension.

    Returns
    -------
    list of str
        The paths written, one per format.
    """
    cols = columns(rows)
    paths = []
    for fmt in formats:
        path = os.path.join(out_dir, f"{filename}.{fmt.lower()}")
        paths.append(write(rows, fmt, path, cols))
    return paths


def format_table(rows: list[Row], units: dict[str, str] | None = None) -> str:
    """Render results as an aligned monospaced table for printing.

    Parameters
    ----------
    rows : list of dict
        Result rows.
    units : dict of str to str or None, optional
        Physical unit for any column, keyed by column name. When a column has a
        unit its header cell is suffixed with ``[unit]`` (e.g.
        ``energy_mae [eV/atom]``). Columns absent from the mapping keep a plain
        header.

    Returns
    -------
    str
        A multi-line string with a header, an underline rule and one line per
        row. Returns ``"(no results)"`` when there are no rows.
    """
    if not rows:
        return "(no results)"
    cols = columns(rows)
    headers = [_unit_label(c, units) for c in cols]
    cells = [[_fmt(r.get(c, "")) for c in cols] for r in rows]
    widths = [max(len(headers[i]), *(len(row[i]) for row in cells))
              for i in range(len(cols))]
    def line(vals):
        return "  ".join(v.ljust(w) for v, w in zip(vals, widths))
    out = [line(headers), line(["-" * w for w in widths])]
    out += [line(row) for row in cells]
    return "\n".join(out)
