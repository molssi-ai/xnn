"""The plain-Python SQLite table reader of the data hub."""
import subprocess
import sys

import numpy as np
import pytest

from xnn.common.data.hub._sqlite import read_table


def _write(path, page_size, rows, sql):
    sqlite3 = pytest.importorskip("sqlite3")   # only to write the test file
    with sqlite3.connect(path) as db:
        db.execute(f"PRAGMA page_size = {page_size}")
        db.execute(sql)
        db.executemany(f"INSERT INTO t VALUES ({','.join('?' * len(rows[0]))})", rows)
    return sqlite3


@pytest.mark.parametrize("page_size", [512, 1024, 4096])
def test_matches_sqlite3_on_every_value_type(tmp_path, page_size):
    """Integers of every width, reals, text, blobs (with overflow pages) and NULL."""
    rng = np.random.default_rng(page_size)
    rows = []
    for i in range(1500):
        big = i % 97 == 0
        rows.append((i + 1, int(rng.integers(-2**62, 2**62)) if i % 3 else int(rng.integers(-300, 300)),
                     float(rng.normal()), None if i % 5 == 0 else f"name {i} é",
                     rng.normal(size=4000 if big else 9).astype("<f4").tobytes(),
                     None if i % 7 == 0 else i % 2))
    path = tmp_path / "t.db"
    sqlite3 = _write(path, page_size, rows,
                     "CREATE TABLE t (id INTEGER NOT NULL PRIMARY KEY, n INTEGER, x REAL, "
                     "s TEXT, b BLOB, k INTEGER)")
    names, got = read_table(path, "t")
    with sqlite3.connect(path) as db:
        want = db.execute("SELECT * FROM t ORDER BY id").fetchall()
    assert names == ["id", "n", "x", "s", "b", "k"]
    assert got == want


def test_rowid_alias_columns_added_later_and_errors(tmp_path):
    """An INTEGER PRIMARY KEY holds the rowid, missing trailing columns read as NULL."""
    path = tmp_path / "t.db"
    sqlite3 = _write(path, 1024, [(5, "a"), (2, "b")],
                     "CREATE TABLE t (id INTEGER PRIMARY KEY, s TEXT, CHECK (length(s) > 0))")
    with sqlite3.connect(path) as db:
        db.execute("ALTER TABLE t ADD COLUMN extra REAL")
    names, rows = read_table(path, "t")
    assert names == ["id", "s", "extra"]
    assert rows == [(2, "b", None), (5, "a", None)]
    with pytest.raises(ValueError, match="no table"):
        read_table(path, "missing")
    other = tmp_path / "not.db"
    other.write_bytes(b"x" * 200)
    with pytest.raises(ValueError, match="not an SQLite"):
        read_table(other, "t")


def test_importing_xnn_data_does_not_load_sqlite3():
    """``sqlite3`` is never imported by the package."""
    code = "import sys, xnn, xnn.common.data; sys.exit(int('sqlite3' in sys.modules))"
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0
