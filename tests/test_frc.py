"""The SEAMM ``.frc`` force-field format (:mod:`xnn.ffnn.common.frc`).

Grammar and resolution are checked on a small synthetic file that exercises
every construct the shipped files use -- ``#define`` with versions and
``:optional`` labels, label-order overriding, ``#include`` (with
``missing_ok``), ``@units`` conversion, ``@type`` nonbond forms, canonical
key ordering, JSON templates with versions, references -- plus the writer's
round trip and the registry of shipped files. The shipped ``oplsaa.frc`` is
then read for a handful of values that can be checked against the file by
eye.
"""
import pytest

from xnn.ffnn.common import (FrcFile, read_frc, find_forcefield,
                              list_forcefields, convert_units,
                              builtin_data_dir)
from xnn.ffnn.common.frc import (canonical_key, nonbond_to_sigma_eps,
                                  parse_version, read_forcefield, make_section)

BASE = """!MolSSI forcefield 1

#version  test.frc  1.0  15-Sep-2026

#define base

!Version  Ref  Function        Label
1.0       1    metadata        base
1.0       1    atom_types      base
1.0       1    equivalence     base
1.0       1    charges         base
1.0       1    nonbond(12-6)   base
1.0       1    quadratic_bond  base
2.0       1    quadratic_bond  base patch extra:optional
1.0       1    torsion_opls    base
1.0       1    improper_opls   base
1.0       1    templates       base
#end

#define patched

!Version  Ref  Function        Label
1.0       1    metadata        base
1.0       1    atom_types      base
1.0       1    equivalence     base
1.0       1    charges         base
1.0       1    nonbond(12-6)   base
1.0       1    quadratic_bond  base patch
1.0       1    torsion_opls    base
1.0       1    templates       base
#end

#include local:__missing__.frc missing_ok

#metadata base
!Version  Ref  Parameter  Value   Description
1.0       1    ff_form    oplsaa  functional form
1.0       1    charges    point   how charges are handled

#atom_types base
!Version  Ref  Type  Mass    El  connections  Comment
1.0       1    c3    12.011  C   4            sp3 carbon
1.0       1    hc    1.008   H   1            hydrogen on carbon
1.0       1    c3m   12.011  C   4            methyl carbon
1.0       1    dm    0.0     X   Dummy        a virtual site

#equivalence base
!Version  Ref  Type  NonB  Bond  Angle  Torsion  OOP
1.0       1    c3    c3    c3    c3     c3       c3
1.0       1    hc    hc    hc    hc     hc       hc
1.0       1    c3m   c3    c3    c3     c3       c3

#charges base
!Version  Ref  I    Q
1.0       1    c3   -0.12
1.1       1    c3   -0.18
1.0       1    hc    0.06
1.0       1    c3m  -0.24

#nonbond(12-6) base
> E = 4 eps [(sigma/r)^12 - (sigma/r)^6]
@type rmin-eps
@units rmin nm
@units eps kJ/mol
@combination geometric
!Version  Ref  I    Rmin       Epsilon
1.0       1    c3   0.392830   0.276144
1.0       1    hc   0.280616   0.125520

#quadratic_bond base
> E = K2 (R - R0)^2
!Version  Ref  I   J   R0     K2
1.0       1    hc  c3  1.090  340.0
1.0       1    c3  c3  1.529  268.0

#quadratic_bond patch
@units R0 nm
@units K2 kJ/mol/nm^2
!Version  Ref  I   J   R0       K2
1.0       1    c3  c3  0.15300  112131.2

#torsion_opls base
!Version  Ref  I   J   K   L   V1    V2     V3   V4
1.0       1    c3  c3  c3  c3  1.3   -0.05  0.2  0.0
1.0       1    *   c3  c3  hc  0.0   0.0    0.3  0.0

#improper_opls base
!Version  Ref  I  J  K   L  V2
1.0       1    *  *  c3  *  2.0

#templates base
{
    "hc": {"1.0": {"smarts": ["[H:1][#6]"], "description": "H on C", "overrides": []}},
    "c3": {"1.0": {"smarts": ["[CX4:1]"], "description": "sp3 C", "overrides": []},
           "1.1": {"smarts": ["[CX4;!H3:1]"], "description": "sp3 C, not methyl", "overrides": []}},
    "c3m": {"1.0": {"smarts": ["[CX4;H3:1]"], "description": "methyl", "overrides": []}}
}

#reference 1
@Author tests
@Date 2026-09-15
Synthetic numbers for the reader tests.

#end
"""


@pytest.fixture
def frc(tmp_path):
    path = tmp_path / "test.frc"
    path.write_text(BASE)
    return read_frc(path)


def test_file_structure(frc):
    assert frc.header == "!MolSSI forcefield 1"
    assert frc.forcefields == ["base", "patched"]
    assert frc.missing_includes == ["local:__missing__.frc"]
    assert ("quadratic_bond", "patch") in frc.sections
    assert set(frc.references) == {(str(frc.path), "1")}
    assert frc.references[(str(frc.path), "1")].author == "tests"


def test_define_versions_and_optional_labels(frc):
    ff = frc.forcefield("base")
    # the newest define row for quadratic_bond lists base + patch (+ an
    # absent optional label), so the patch overrides the base c3-c3 bond
    assert ff.labels["quadratic_bond"] == ["base", "patch"]
    assert ff.bond("c3", "c3")[2].values["K2"] == pytest.approx(268.0)
    # ... and the version-1.0 view of the same force field ignores the patch
    ff1 = frc.forcefield("base", version="1.0")
    assert ff1.labels["quadratic_bond"] == ["base"]
    # a non-optional missing label is an error
    frc.defines["base"].entries.append(("3.0", "1", "charges", ["nope"]))
    with pytest.raises(KeyError):
        frc.forcefield("base")
    with pytest.raises(KeyError):
        frc.forcefield("does-not-exist")


def test_units_are_converted_to_the_schema_defaults(frc):
    ff = frc.forcefield("base")
    row = ff.rows("quadratic_bond")[("c3", "c3")]
    # 0.153 nm -> 1.53 A ; 112131.2 kJ/mol/nm^2 -> 268.0 kcal/mol/A^2 (the
    # .frc form is K2 (R-R0)^2 with no 1/2, unlike GROMACS' kb)
    assert row.values["R0"] == pytest.approx(1.53)
    assert row.values["K2"] == pytest.approx(268.0, rel=1e-6)
    assert convert_units(1.0, "kJ/mol", "kcal/mol") == pytest.approx(1 / 4.184)
    assert convert_units(180.0, "degree", "radian") == pytest.approx(3.141592653589793)
    assert convert_units(2.0, "(kJ/mol)**(1/6)*nm", "(kcal/mol)**(1/6)*Å") \
        == pytest.approx(2.0 * (1 / 4.184) ** (1 / 6) * 10.0)
    with pytest.raises(ValueError):
        convert_units(1.0, "kJ/mol", "nm")
    with pytest.raises(ValueError):
        convert_units(1.0, "furlong", "nm")


def test_nonbond_forms_are_normalised_at_read(frc):
    ff = frc.forcefield("base")
    sig, eps = ff.nonbond("c3")
    # rmin 0.39283 nm -> sigma = 3.9283 / 2^(1/6) A ; 0.276144 kJ -> 0.066 kcal
    assert sig == pytest.approx(3.9283 / 2 ** (1 / 6), rel=1e-6)
    assert eps == pytest.approx(0.066, rel=1e-6)
    assert ff.nonbond("c3m") == ff.nonbond("c3")        # via NonB equivalence
    assert ff.nonbond("dm") is None
    assert ff.combination() == "geometric"
    assert ff.modifiers["nonbond(12-6)"]["type"] == [["sigma-eps"]]
    assert "units" not in ff.modifiers["nonbond(12-6)"]
    assert "units" not in ff.modifiers["quadratic_bond"]   # converted on read
    # the transforms themselves
    assert nonbond_to_sigma_eps("A-B", 4.0 * 1.0 * 2.0 ** 12, 4.0 * 1.0 * 2.0 ** 6) \
        == pytest.approx((2.0, 1.0))
    a_r, b_r = (4.0 * 2.0 ** 12) ** (1 / 12), (4.0 * 2.0 ** 6) ** (1 / 6)
    assert nonbond_to_sigma_eps("A/r-B/r", a_r, b_r) == pytest.approx((2.0, 1.0))
    assert nonbond_to_sigma_eps("eps-rmin", 1.0, 2.0 ** (1 / 6) * 3.0) \
        == pytest.approx((3.0, 1.0))


def test_versions_charges_and_equivalences(frc):
    ff = frc.forcefield("base")
    assert ff.charge("c3") == pytest.approx(-0.18)     # newest version wins
    assert ff.charge("c3m") == pytest.approx(-0.24)   # its own row wins
    assert ff.charge("dm") == 0.0                      # no row: default zero
    assert ff.equivalent("c3m", "bond") == "c3"
    assert ff.equivalent("c3m", "nonbond") == "c3"
    assert ff.equivalent("dm", "torsion") == "dm"      # no equivalence row
    assert ff.metadata == {"ff_form": "oplsaa", "charges": "point"}
    assert ff.atom_types["dm"]["connections"] == "Dummy"
    assert ff.atom_types["c3"]["Comment"] == "sp3 carbon"
    assert ff.atom_types["c3"]["Mass"] == pytest.approx(12.011)


def test_bonded_lookups_with_equivalences_and_wildcards(frc):
    ff = frc.forcefield("base")
    assert ff.bond("c3m", "hc")[1] == ("c3", "hc")      # equivalence + canonical order
    assert ff.bond("hc", "c3m")[1] == ("c3", "hc")
    assert ff.bond("dm", "dm") is None
    t = ff.torsion("c3m", "c3", "c3", "c3m")
    assert t[1] == ("c3", "c3", "c3", "c3") and t[2].values["V1"] == 1.3
    t = ff.torsion("hc", "c3", "c3", "c3m")               # only a wildcard row
    assert t[1] == ("*", "c3", "c3", "hc") and t[2].values["V3"] == 0.3
    t = ff.torsion("c3m", "c3", "c3", "hc")               # reversed spelling
    assert t[1] == ("*", "c3", "c3", "hc")
    im = ff.improper("hc", "hc", "c3m", "c3")
    assert im[1] == ("*", "*", "c3", "*") and im[2].values["V2"] == 2.0
    assert ff.improper("hc", "hc", "hc", "hc") is None
    assert ff.terms["bond"] == ["quadratic_bond"]


def test_templates_keep_newest_version_in_file_order(frc):
    ff = frc.forcefield("base")
    assert list(ff.templates) == ["hc", "c3", "c3m"]
    assert ff.templates["c3"]["smarts"] == ["[CX4;!H3:1]"]
    assert ff.templates["c3"]["version"] == "1.1"
    assert frc.forcefield("base", version="1.0").templates["c3"]["smarts"] \
        == ["[CX4:1]"]


def test_canonical_keys():
    assert canonical_key("like_bond", ("b", "a")) == (("a", "b"), True)
    assert canonical_key("like_angle", ("c", "x", "a")) == (("a", "x", "c"), True)
    assert canonical_key("like_torsion", ("d", "c", "b", "a")) \
        == (("a", "b", "c", "d"), True)
    assert canonical_key("like_torsion", ("d", "b", "b", "a")) \
        == (("a", "b", "b", "d"), True)
    assert canonical_key("like_improper", ("c", "a", "k", "b")) \
        == (("a", "b", "k", "c"), True)
    assert canonical_key("like_oop", ("c", "j", "a", "b")) \
        == (("a", "j", "b", "c"), True)
    assert canonical_key("none", ("b", "a")) == (("b", "a"), False)
    assert parse_version("2023.08.21") > parse_version("2023.01.29") \
        > parse_version("2.1") > parse_version("1.0")


def test_duplicate_section_and_define_are_errors(tmp_path):
    bad = tmp_path / "dup.frc"
    bad.write_text(BASE + "\n#charges base\n!Version Ref I Q\n1.0 1 hc 0.1\n")
    with pytest.raises(ValueError, match="more than once"):
        read_frc(bad)
    bad.write_text(BASE + "\n#define base\n!V R F L\n1.0 1 charges base\n#end\n")
    with pytest.raises(ValueError, match="defined twice"):
        read_frc(bad)
    missing = tmp_path / "inc.frc"
    missing.write_text("!MolSSI forcefield 1\n#include nowhere.frc\n#end\n")
    with pytest.raises(FileNotFoundError):
        read_frc(missing)


def test_include_splices_sections(tmp_path):
    (tmp_path / "part.frc").write_text(
        "!MolSSI forcefield 1\n#charges extra\n!Version Ref I Q\n"
        "1.0 1 zz 0.5\n#end\n")
    (tmp_path / "main.frc").write_text(
        "!MolSSI forcefield 1\n#define m\n!V R F L\n1.0 1 charges extra\n#end\n"
        "#include part.frc\n#end\n")
    ff = read_frc(tmp_path / "main.frc").forcefield()
    assert ff.charge("zz") == 0.5
    # local: resolves against include_dirs, then the xnn data directory
    (tmp_path / "main2.frc").write_text(
        "!MolSSI forcefield 1\n#define m\n!V R F L\n1.0 1 charges extra\n#end\n"
        "#include local:part.frc\n#end\n")
    ff2 = read_frc(tmp_path / "main2.frc", include_dirs=[tmp_path]).forcefield()
    assert ff2.charge("zz") == 0.5


def test_writer_round_trip(frc, tmp_path):
    out = tmp_path / "written.frc"
    frc.write(out)
    back = read_frc(out)
    assert back.forcefields == frc.forcefields
    a, b = frc.forcefield("patched"), back.forcefield("patched")
    for kind in a.sections:
        assert set(a.rows(kind)) == set(b.rows(kind)), kind
        for key, row in a.rows(kind).items():
            for col, val in row.values.items():
                got = b.rows(kind)[key].values[col]
                if isinstance(val, float):
                    assert got == pytest.approx(val, rel=1e-9), (kind, key, col)
                else:
                    assert got == val, (kind, key, col)
    assert b.templates == a.templates
    assert b.nonbond("c3") == pytest.approx(a.nonbond("c3"))
    # hand-built sections through make_section canonicalise their keys
    sec = make_section("quadratic_bond", "x", ["I", "J"], ["R0", "K2"],
                       [(("b", "a"), {"R0": 1.0, "K2": 2.0})])
    assert sec.rows[0].key == ("a", "b")


def test_registry_and_spec_forms():
    known = list_forcefields()
    for name in ("oplsaa", "CL&P", "oplsaa+", "lopls", "oplsaa-1996",
                 "reaxff/CHO_cho_2008"):
        assert name in known
    assert find_forcefield("oplsaa")[1] == "oplsaa"
    assert find_forcefield("CHO_cho_2008")[1] == "reaxff/CHO_cho_2008"
    assert find_forcefield("cho_cho_2008")[1] == "reaxff/CHO_cho_2008"
    path = builtin_data_dir() / "oplsaa.frc"
    assert find_forcefield(f"{path}:CL&P") == (path, "CL&P")
    assert find_forcefield(str(path)) == (path, None)
    with pytest.raises(FileNotFoundError):
        find_forcefield("no-such-force-field")
    with pytest.raises(FileNotFoundError):
        find_forcefield("/no/such/file.frc")


def test_shipped_oplsaa_values():
    """Spot checks against the shipped oplsaa.frc (readable by eye in the file)."""
    ff = read_forcefield("oplsaa")
    assert ff.ff_form == "oplsaa" and len(ff.templates) == 572
    assert ff.bond("opls_18", "opls_18")[2].values == {"R0": 1.529, "K2": 268.0}
    assert ff.angle("opls_18", "opls_18", "opls_18")[2].values \
        == {"Theta0": 112.7, "K2": 58.35}
    assert ff.torsion("opls_18", "opls_18", "opls_18", "opls_18")[2].values \
        == {"V1": 1.3, "V2": -0.05, "V3": 0.2, "V4": 0.0}
    assert ff.charge("opls_80") == -0.18 and ff.nonbond("opls_80") == (3.5, 0.066)
    # alkane CH3 carbon opls_80 bonds through its equivalent opls_18
    assert ff.bond("opls_80", "opls_81")[1] == ("opls_18", "opls_18")
    # the CL&P-g bonds are given in nm / kJ and come out in A / kcal
    plus = read_forcefield("oplsaa+")
    row = plus.rows("quadratic_bond")[("B", "Fbob")]
    assert row.values["R0"] == pytest.approx(1.382)
    assert row.values["K2"] == pytest.approx(323500.0 / 4.184 / 100.0)
    # oplsaa-1996 layers its torsions over oplsaa through #include
    f96 = read_forcefield("oplsaa-1996")
    assert f96.torsion("opls_18", "opls_18", "opls_18", "opls_18")[2].values["V1"] \
        == 1.74
    assert f96.labels["torsion_opls"] == ["oplsaa", "oplsaa-1996"]
    assert len(f96.templates) == 572
