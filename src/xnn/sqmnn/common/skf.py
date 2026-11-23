"""Slater-Koster files of density-functional tight binding: reader and published sets.

A Slater-Koster file ``A-B.skf`` holds the two-center Hamiltonian and overlap
integrals between the valence orbitals of an atom of element ``A`` and one of
element ``B`` on a uniform grid of distances (bohr), plus the repulsive pair
potential. Files of one element with itself (``A-A.skf``) also hold the atomic
orbital energies, Hubbard parameters and occupations.

Layout of a file:

* first line: grid spacing and number of grid points (an ``@`` line before it
  marks the extended format with f orbitals);
* homonuclear files only: orbital energies, spin-polarization energy, Hubbard
  parameters and occupations, from the highest angular momentum down;
* atomic mass, repulsive polynomial coefficients ``c2 ... c9`` and its cutoff;
* one row per grid point, starting at one grid spacing: the Hamiltonian
  integrals then the overlap integrals, ten each (twenty in the extended
  format), from the highest angular momenta down to ``ss sigma``;
* optionally a ``Spline`` block: the repulsive potential as an exponential
  head, cubic segments and a fifth-order last segment.

Numbers follow Fortran list-directed input: commas separate like blanks and
``n*v`` repeats ``v`` ``n`` times.

xnn does not ship Slater-Koster files. The published sets (github.com/dftbparams,
Creative Commons Attribution-ShareAlike 4.0) are downloaded at a fixed commit on
first use into the model cache under ``sqmnn/dftb/<set>``, checked against a
digest of their contents, and read from there afterwards. Their README files
name the papers that must be cited when a set is used. With ``XNN_OFFLINE=1``
the set must already be cached. A folder of files in the same format can be
used instead.
"""
from __future__ import annotations

import hashlib
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from xnn.common.data.elements import CHEMICAL_SYMBOLS

#: Integral columns of one matrix in a row, simple format (``l1 <= l2``, highest first).
COLUMNS = ("dd0", "dd1", "dd2", "pd0", "pd1", "pp0", "pp1", "sd0", "sp0", "ss0")
#: The same for the extended format with f orbitals.
EXTENDED_COLUMNS = ("ff0", "ff1", "ff2", "ff3", "df0", "df1", "df2", "dd0", "dd1", "dd2", "pf0", "pf1",
                    "pd0", "pd1", "pp0", "pp1", "sf0", "sd0", "sp0", "ss0")


@dataclass
class Spline:
    """Repulsive potential of a ``Spline`` block (bohr, hartree).

    Below the first knot it is ``exp(-a1 r + a2) + a3``; between knots ``r0`` and ``r1`` it is
    ``sum_k c_k (r - r0)^k`` (cubic, the last segment fifth order); it vanishes beyond ``cutoff``.
    """

    cutoff: float
    head: Tuple[float, float, float]
    knots: np.ndarray                                          # (n + 1,) segment boundaries
    coefficients: np.ndarray                                   # (n, 6), cubic segments padded with zeros


@dataclass
class SlaterKosterFile:
    """The contents of one Slater-Koster file."""

    grid: float                                                # spacing of the table (bohr)
    n_grid: int                                                # number of points the first line states
    hamiltonian: np.ndarray                                    # (rows, n_columns), row k at (k + 1) * grid
    overlap: np.ndarray
    extended: bool = False
    mass: float = 0.0
    polynomial: Tuple[float, ...] = ()                         # c2 ... c9
    polynomial_cutoff: float = 0.0
    spline: Optional[Spline] = None
    #: Homonuclear files: values by angular momentum ``l = 0, 1, 2(, 3)``.
    energies: Optional[Tuple[float, ...]] = None
    hubbard: Optional[Tuple[float, ...]] = None
    occupations: Optional[Tuple[float, ...]] = None
    spin_polarization_energy: float = 0.0
    #: Number of shells the first line states (some sets add it after the grid).
    n_shells: Optional[int] = None
    #: Range separation of the set (``("LC", omega)``, omega in inverse bohr), from a ``RangeSep`` block.
    range_separation: Optional[Tuple[str, float]] = None

    @property
    def columns(self) -> Tuple[str, ...]:
        """Names of the integral columns of each matrix."""
        return EXTENDED_COLUMNS if self.extended else COLUMNS


def _numbers(line: str, strict: bool = True) -> List[float]:
    """The numbers of one line of Fortran list-directed input; with ``strict=False`` tokens that are not
    numbers (logical flags some sets append) end the line."""
    out: List[float] = []
    for token in line.replace(",", " ").split():
        try:
            if "*" in token:
                count, value = token.split("*", 1)
                out.extend([float(value.replace("d", "e").replace("D", "e"))] * int(count))
            else:
                out.append(float(token.replace("d", "e").replace("D", "e")))
        except ValueError:
            if strict:
                raise
            break
    return out


def parse_skf(text: str, homonuclear: bool) -> SlaterKosterFile:
    """Read the text of a Slater-Koster file.

    Parameters
    ----------
    text : str
        The file contents.
    homonuclear : bool
        The file couples an element with itself (it has the line of atomic values).

    Returns
    -------
    SlaterKosterFile
    """
    lines = [line for line in text.splitlines() if line.strip()]
    pos = 0
    extended = lines[0].strip().startswith("@")
    if extended:
        pos += 1
    first = _numbers(lines[pos])
    grid, n_grid = first[0], int(first[1])
    n_shells = int(first[2]) if len(first) > 2 else None
    pos += 1
    n_l = 4 if extended else 3
    energies = hubbard = occupations = None
    spe = 0.0
    if homonuclear:
        v = _numbers(lines[pos], strict=False)
        pos += 1
        energies = tuple(v[:n_l][::-1])
        spe = v[n_l]
        hubbard = tuple(v[n_l + 1:2 * n_l + 1][::-1])
        occupations = tuple(v[2 * n_l + 1:3 * n_l + 1][::-1])
    v = _numbers(lines[pos], strict=False)
    pos += 1
    mass, polynomial, rcut = v[0], tuple(v[1:9]), v[9]
    n_columns = len(EXTENDED_COLUMNS if extended else COLUMNS)
    rows = []
    while pos < len(lines) and not re.match(r"\s*(Spline|<|#)", lines[pos]):
        try:
            values = _numbers(lines[pos])
        except ValueError:                      # text after the table (comments, fitting data)
            break
        if len(values) != 2 * n_columns:
            break
        rows.append(values)
        pos += 1
    table = np.asarray(rows, dtype=np.float64).reshape(-1, 2 * n_columns)
    spline = None
    while pos < len(lines) and not lines[pos].strip().startswith(("Spline", "<")):
        pos += 1                                # comments or other blocks between the table and the spline
    if pos < len(lines) and lines[pos].strip().startswith("Spline"):
        n_seg, cutoff = _numbers(lines[pos + 1])[:2]
        n_seg = int(n_seg)
        head = tuple(_numbers(lines[pos + 2])[:3])
        knots, coefficients = [], np.zeros((n_seg, 6))
        for k in range(n_seg):
            seg = _numbers(lines[pos + 3 + k])
            knots.append(seg[0])
            coefficients[k, :len(seg) - 2] = seg[2:8]
            if k == n_seg - 1:
                knots.append(seg[1])
        spline = Spline(float(cutoff), head, np.asarray(knots), coefficients)
    range_separation = None
    for k in range(pos, len(lines) - 1):
        if lines[k].strip() == "RangeSep":
            kind, *values = lines[k + 1].split()
            range_separation = (kind, float(values[0]))
            break
        if lines[k].lstrip().startswith("<"):
            break
    return SlaterKosterFile(grid=grid, n_grid=n_grid, hamiltonian=table[:, :n_columns].copy(),
                            overlap=table[:, n_columns:].copy(), extended=extended, mass=mass,
                            polynomial=polynomial, polynomial_cutoff=rcut, spline=spline, energies=energies,
                            hubbard=hubbard, occupations=occupations, spin_polarization_energy=spe,
                            n_shells=n_shells, range_separation=range_separation)


def format_skf(f: SlaterKosterFile) -> str:
    """The text of a Slater-Koster file (numbers in their shortest exact form); :func:`parse_skf` reads it back
    unchanged."""
    num = lambda x: repr(float(x))  # noqa: E731
    lines = ["@ written by xnn"] if f.extended else []
    lines.append(f"{num(f.grid)} {f.n_grid}" + (f" {f.n_shells}" if f.n_shells else ""))
    n_l = 4 if f.extended else 3
    if f.energies is not None:
        pad = lambda v: list(v) + [0.0] * (n_l - len(v))  # noqa: E731
        values = pad(f.energies)[::-1] + [f.spin_polarization_energy] + pad(f.hubbard)[::-1] + pad(f.occupations)[::-1]
        lines.append(" ".join(num(x) for x in values))
    lines.append(" ".join(num(x) for x in [f.mass, *f.polynomial, f.polynomial_cutoff] + [0.0] * 10))
    for h, o in zip(f.hamiltonian, f.overlap):
        lines.append(" ".join(num(x) for x in list(h) + list(o)))
    if f.spline is not None:
        sp = f.spline
        n = sp.coefficients.shape[0]
        lines += ["Spline", f"{n} {num(sp.cutoff)}", " ".join(num(x) for x in sp.head)]
        for k in range(n):
            width = 6 if k == n - 1 else 4
            lines.append(" ".join(num(x) for x in [sp.knots[k], sp.knots[k + 1], *sp.coefficients[k, :width]]))
    return "\n".join(lines) + "\n"


@dataclass
class SlaterKosterSet:
    """A folder of Slater-Koster files named ``<A>-<B>.skf``, read on demand."""

    folder: Path
    name: str = ""
    _files: Dict[Tuple[int, int], SlaterKosterFile] = field(default_factory=dict, repr=False)

    def path(self, za: int, zb: int) -> Path:
        """The file of the element pair ``(za, zb)``."""
        return Path(self.folder) / f"{CHEMICAL_SYMBOLS[za]}-{CHEMICAL_SYMBOLS[zb]}.skf"

    def has(self, za: int, zb: int) -> bool:
        """Whether the set has the file of the pair."""
        return self.path(za, zb).exists()

    def pair(self, za: int, zb: int) -> SlaterKosterFile:
        """The parsed file of the pair ``(za, zb)`` (orbitals of ``za`` first)."""
        key = (za, zb)
        if key not in self._files:
            path = self.path(za, zb)
            if not path.exists():
                raise KeyError(f"the Slater-Koster set {self.name or self.folder} has no file {path.name}")
            self._files[key] = parse_skf(path.read_text(), za == zb)
        return self._files[key]

    def readme(self) -> str:
        """The README of the set (the folder's or its parent's), empty when there is none."""
        for folder in (Path(self.folder), Path(self.folder).parent, Path(self.folder).parent.parent):
            if (folder / "README").exists():
                return (folder / "README").read_text(errors="replace")
        return ""

    def recommended(self) -> Dict[str, Dict[int, Union[float, str]]]:
        """Settings the README recommends, by name: ``"max_angular_momentum"`` (letter per atomic
        number), ``"hubbard_derivatives"`` (hartree per electron) and ``"damping_exponent"`` (key 0)."""
        text = self.readme()
        out: Dict[str, Dict[int, Union[float, str]]] = {"max_angular_momentum": {}, "hubbard_derivatives": {},
                                                        "damping_exponent": {}}
        number = {s: z for z, s in enumerate(CHEMICAL_SYMBOLS)}
        block = re.search(r"Maximum angular momenta[^\n]*\n(.*?)(?:\n-{5,}|\Z)", text, re.S)
        if block:
            for el, l in re.findall(r"^\s*([A-Z][a-z]?)\s*=\s*\"([spdf])\"", block.group(1), re.M):
                out["max_angular_momentum"][number[el]] = l
        block = re.search(r"Hubbard derivatives[^\n]*\n(.*?)(?:\n-{5,}|\Z)", text, re.S)
        if block:
            for el, v in re.findall(r"^\s*([A-Z][a-z]?)\s*=\s*([-+.\deE]+)\s*$", block.group(1), re.M):
                out["hubbard_derivatives"][number[el]] = float(v)
        m = re.search(r"^\s*zeta\s*=\s*([-+.\deE]+)", text, re.M)
        if m:
            out["damping_exponent"][0] = float(m.group(1))
        return out

    def elements(self) -> Tuple[int, ...]:
        """Atomic numbers with a homonuclear file."""
        out = []
        for z, symbol in enumerate(CHEMICAL_SYMBOLS):
            if z and (Path(self.folder) / f"{symbol}-{symbol}.skf").exists():
                out.append(z)
        return tuple(out)


@dataclass(frozen=True)
class _Published:
    repository: str
    commit: str
    digest: str                                                # MD5 over "<md5>  <path>" lines of the files
    folder: str = "skfiles"


#: The published sets, by name: repository under github.com/dftbparams, pinned commit, content digest.
PUBLISHED: Dict[str, _Published] = {
    "mio-1-1": _Published("mio", "263e52419ce15222c864df9b3f6fbf58bfbdfa0f", "666f83c43fadfa2e22a1458153259833"),
    "3ob-3-1": _Published("3ob", "c5e165cb65f80f6b4e054c99e3f770ac3b8a4ecc", "31a5d9f74cc8fd65b8eecff3227ac4d0"),
    "ob2-1-1": _Published("ob2", "3eef8736ac852be619a8e18cc91976e3c44d3768", "cc0f90ea0ea6fdb9720883498153016f"),
    "pbc-0-3": _Published("pbc", "67b5ecb421d0e8b64be0b0ecb7373a832e04dd3f", "0c01a3d57bd05f92ee788e00815af2d1"),
    "matsci-0-3": _Published("matsci", "57016b4363fd6180f6edff662e5fbaa95276b4bd",
                             "0045a6029810500b4950b50ce97db5f4"),
}
#: Short names accepted for the published sets.
ALIASES = {"mio": "mio-1-1", "3ob": "3ob-3-1", "ob2": "ob2-1-1", "pbc": "pbc-0-3", "matsci": "matsci-0-3"}
#: Sets with variants in subfolders: the default variant.
VARIANTS = {"ob2-1-1": ("base", ("base", "shift", "split"))}


def content_digest(folder: Path) -> str:
    """MD5 over the lines ``"<md5>  <relative path>"`` of every ``.skf`` file under ``folder``, sorted."""
    folder = Path(folder)
    lines = []
    for path in sorted(folder.rglob("*.skf"), key=lambda p: p.relative_to(folder).as_posix().encode()):
        lines.append(f"{hashlib.md5(path.read_bytes()).hexdigest()}  {path.relative_to(folder).as_posix()}\n")
    return hashlib.md5("".join(lines).encode()).hexdigest()


def published_folder(name: str, cache_dir: Optional[Union[str, Path]] = None) -> Path:
    """The folder of a published set in the model cache, downloaded on first use.

    Parameters
    ----------
    name : str
        A key of :data:`PUBLISHED` or :data:`ALIASES`, optionally with a variant (``"ob2/split"``).
    cache_dir : str or Path, optional
        Model cache; defaults to the xnn model cache.

    Returns
    -------
    Path
        The folder that holds the ``.skf`` files.

    Raises
    ------
    FileNotFoundError
        When the set is not cached and ``XNN_OFFLINE`` forbids downloading.
    """
    from xnn.common.data.hub._download import download_file, extract_archive
    from xnn.common.models.hub.cache import offline, resolve_cache_dir
    from xnn.common.models.hub.provenance import write_license_note

    base, _, variant = name.strip().lower().partition("/")
    key = ALIASES.get(base, base)
    if key not in PUBLISHED:
        raise KeyError(f"unknown Slater-Koster set {name!r}; published sets: {sorted(PUBLISHED)}")
    pub = PUBLISHED[key]
    if key in VARIANTS:
        default, allowed = VARIANTS[key]
        variant = variant or default
        if variant not in allowed:
            raise KeyError(f"the set {key} has the variants {list(allowed)}, not {variant!r}")
    elif variant:
        raise KeyError(f"the set {key} has no variants")
    root = resolve_cache_dir(cache_dir) / "sqmnn" / "dftb" / key
    files = root / pub.folder
    source = f"https://github.com/dftbparams/{pub.repository}/tree/{pub.commit}"
    note = (root, f"the DFTB parameter set {key}", pub.commit[:12], source, "CC-BY-SA-4.0",
            (f"https://raw.githubusercontent.com/dftbparams/{pub.repository}/{pub.commit}/LICENSE",))
    if not (files.is_dir() and content_digest(files) == pub.digest):
        if offline():
            raise FileNotFoundError(
                f"{files} is not cached and XNN_OFFLINE is set; load the set once with network access, "
                "or pass a folder of Slater-Koster files")
        archive = root.parent / f"{key}-{pub.commit[:12]}.tar.gz"
        download_file(f"https://codeload.github.com/dftbparams/{pub.repository}/tar.gz/{pub.commit}", archive,
                      quiet=True)
        if files.exists():
            shutil.rmtree(files)
        keep = (pub.folder + "/", "README", "LICENSE")
        extract_archive(archive, root, select=lambda m: m.split("/", 1)[-1].startswith(keep), strip=1)
        archive.unlink(missing_ok=True)
        got = content_digest(files)
        if got != pub.digest:
            raise ValueError(f"the downloaded set {key} does not match its pinned contents ({got} != {pub.digest})")
    write_license_note(*note)
    return files / variant if variant else files


def published_set(name: str, cache_dir: Optional[Union[str, Path]] = None) -> SlaterKosterSet:
    """A published Slater-Koster set (see :func:`published_folder`)."""
    return SlaterKosterSet(published_folder(name, cache_dir), name=name)


def load_set(source: Union[str, Path, SlaterKosterSet], cache_dir: Optional[Union[str, Path]] = None
             ) -> SlaterKosterSet:
    """A set from a published name, a folder of files, or an existing set."""
    if isinstance(source, SlaterKosterSet):
        return source
    text = str(source)
    if Path(text).is_dir():
        return SlaterKosterSet(Path(text), name=Path(text).name)
    return published_set(text, cache_dir)
