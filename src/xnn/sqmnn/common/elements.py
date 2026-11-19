"""Element data of the NDDO methods (MNDO family: PM3, PM6, PM6-ORG, PM7).

Tables indexed by atomic number (index 0 is a dummy) for Z = 1..86. They
hold what the methods define per element besides their fitted parameters:

* the reference valence-shell occupations of the isolated atom (``s``, ``p``,
  ``d`` electrons), whose sum is the core charge and which fix the one-center
  energy of the isolated atom;
* the principal quantum numbers of the valence ``s``, ``p`` and ``d`` Slater
  orbitals (Table 1 of the PM6 paper; the rare gases Ne-Xe use the shells
  ``np`` and ``(n+1)s``);
* the row of the periodic table, which sets the principal quantum number of
  the multipole charge separations;
* whether the one-center ``sp`` two-electron integrals are fitted parameters
  (main group, Zn, Cd, Hg) or follow from the internal orbital exponents
  (transition metals);
* the experimental heats of formation of the gaseous atoms (kcal/mol);
* covalent radii (Angstrom) for the bonding topology, average atomic masses
  for the center of mass of ions.
"""
from __future__ import annotations

from ...common.data.elements import CHEMICAL_SYMBOLS, atomic_number  # noqa: F401

MAX_Z = 86

# valence occupations of the isolated atoms
_S = ([1, 2]
      + [1, 2, 2, 2, 2, 2, 2, 0]
      + [1, 2, 2, 2, 2, 2, 2, 0]
      + [1, 2, 2, 2, 2, 1, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 0]
      + [1, 2, 2, 2, 1, 1, 2, 1, 1, 0, 1, 2, 2, 2, 2, 2, 2, 0]
      + [1, 2, 2] + [2] * 14 + [2, 2, 1, 2, 2, 2, 1, 1, 2, 2, 2, 2, 2, 2, 0])
_P = ([0, 0]
      + [0, 0, 1, 2, 3, 4, 5, 6]
      + [0, 0, 1, 2, 3, 4, 5, 6]
      + [0] * 12 + [1, 2, 3, 4, 5, 6]
      + [0] * 12 + [1, 2, 3, 4, 5, 6]
      + [0] * 26 + [1, 2, 3, 4, 5, 6])
_D = ([0, 0] + [0] * 8 + [0] * 8
      + [0, 0, 1, 2, 3, 5, 5, 6, 7, 8, 10, 0] + [0] * 6
      + [0, 0, 1, 2, 4, 5, 5, 7, 8, 10, 10, 0] + [0] * 6
      + [0, 0, 1] + [1] * 13 + [1, 2, 3, 5, 5, 6, 7, 9, 10, 0] + [0] * 6)
#: Reference ``s`` occupation of the isolated atom.
OCC_S = (0,) + tuple(_S)
#: Reference ``p`` occupation of the isolated atom.
OCC_P = (0,) + tuple(_P)
#: Reference ``d`` occupation of the isolated atom.
OCC_D = (0,) + tuple(_D)
#: Core charge (number of valence electrons of the neutral atom).
CORE_CHARGE = tuple(s + p + d for s, p, d in zip(OCC_S, OCC_P, OCC_D))

# principal quantum numbers of the valence Slater orbitals
_PQN_S = ([1, 1] + [2] * 7 + [3] + [3] * 7 + [4] + [4] * 17 + [5]
          + [5] * 17 + [6] + [6] * 31 + [7])
_PQN_P = ([1, 2] + [2] * 8 + [3] * 8 + [4] * 18 + [5] * 18 + [6] * 32)
_PQN_D = ([0, 0] + [0] * 8 + [3] * 7 + [4] + [3] * 12 + [4] * 5 + [5]
          + [4] * 12 + [5] * 5 + [6] + [5] * 26 + [6] * 5 + [7])
#: Principal quantum number of the valence ``s`` orbital.
PQN_S = (0,) + tuple(_PQN_S)
#: Principal quantum number of the valence ``p`` orbitals.
PQN_P = (0,) + tuple(_PQN_P)
#: Principal quantum number of the valence ``d`` orbitals.
PQN_D = (0,) + tuple(_PQN_D)


def period(z: int) -> int:
    """Row of the periodic table (1 for H and He, ..., 6 for Cs-Rn)."""
    for row, last in enumerate((2, 10, 18, 36, 54, 86), start=1):
        if z <= last:
            return row
    raise ValueError(f"no NDDO data beyond Z = {MAX_Z}")


def d_shell(z: int) -> int:
    """Principal quantum number assigned to ``d`` charge distributions (3, 4, 5, or 6 for the p block of the sixth
    row, where only PM7's bismuth has a ``d`` shell); the ``d`` orbital's own number of :data:`PQN_D`."""
    return 3 if z <= 30 else (4 if z <= 48 else (5 if z <= 80 else 6))


#: ``True`` where the one-center ``sp`` integrals (GSS, GSP, GPP, GP2, HSP) are
#: parameters; ``False`` for the transition metals, whose integrals follow from
#: the internal exponents ZSN and ZPN.
MAIN_GROUP = (False,) + tuple(not (21 <= z <= 29 or 39 <= z <= 47 or 57 <= z <= 79)
                              for z in range(1, MAX_Z + 1))

#: Experimental heats of formation of the gaseous atoms, kcal/mol.
ATOM_HEAT = (0.0,
             52.102, 0.000, 38.410, 76.960, 135.700, 170.890, 113.000, 59.559, 18.890, 0.000,
             25.650, 35.000, 79.490, 108.390, 75.570, 66.400, 28.990, 0.000, 21.420, 42.600,
             90.300, 112.300, 122.900, 95.000, 67.700, 99.300, 102.400, 102.800, 80.700, 31.170,
             65.400, 89.500, 72.300, 54.300, 26.740, 0.000, 19.600, 39.100, 101.500, 145.500,
             172.400, 157.300, 162.000, 155.500, 133.000, 90.000, 68.100, 26.720, 58.000, 72.200,
             63.200, 47.000, 25.517, 0.000, 18.700, 42.500, 103.011, 101.004, 84.990, 78.298,
             83.174, 49.402, 41.898, 95.007, 92.902, 69.407, 71.893, 75.791, 55.500, 36.358,
             102.199, 148.000, 186.900, 203.100, 185.000, 188.000, 160.000, 135.200, 88.000,
             14.690, 43.550, 46.620, 50.100, 0.0, 0.0, 0.000)

#: Covalent radii (Angstrom) of the bonding topology; ``None`` where undefined.
COVALENT_RADIUS = (0.0,
                   0.37, 0.32, 1.34, 0.90, 0.82, 0.77, 0.75, 0.73, 0.71, 0.69,
                   1.54, 1.30, 1.18, 1.11, 1.06, 1.02, 0.99, 0.97, 1.96, 1.74,
                   1.44, 1.36, 1.25, 1.27, 1.39, 1.25, 1.26, 1.21, 1.38, 1.31,
                   1.26, 1.22, 1.19, 1.16, 1.14, 1.10, 2.11, 1.92, 1.62, 1.48,
                   1.37, 1.45, 1.56, 1.26, 1.35, 1.31, 1.53, 1.48, 1.44, 1.41,
                   1.38, 1.35, 1.33, 1.30, 2.25, 1.98, 1.69, 1.0, 1.0, 1.0,
                   1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                   1.60, 1.50, 1.38, 1.46, 1.59, 1.28, 1.37, 1.28, 1.44, 1.49,
                   1.48, 1.47, 1.46, None, None, 1.45)

#: Alkali and alkaline-earth metals: they never count as bonded in the topology.
METALS = frozenset((3, 11, 12, 19, 20, 37, 38, 55, 56))

#: Average atomic masses (u), for the center of mass of charged species.
ATOMIC_MASS = (0.0,
               1.00790, 4.00260, 6.94000, 9.01218, 10.81000, 12.01100, 14.00670, 15.99940,
               18.99840, 20.17900, 22.98977, 24.30500, 26.98154, 28.08550, 30.97376, 32.06000,
               35.45300, 39.94800, 39.09830, 40.08000, 44.95590, 47.90000, 50.94150, 51.99600,
               54.93800, 55.84700, 58.93320, 58.71000, 63.54600, 65.38000, 69.73500, 72.59000,
               74.92160, 78.96000, 79.90400, 83.80000, 85.46780, 87.62000, 88.90590, 91.22000,
               92.90640, 95.94000, 98.90620, 101.0700, 102.9055, 106.4000, 107.8680, 112.4100,
               114.8200, 118.6900, 121.7500, 127.6000, 126.9045, 131.3000, 132.9054, 137.3300,
               138.9060, 140.1160, 140.9077, 144.2400, 145.0000, 150.3600, 151.9640, 157.2500,
               158.9253, 162.5000, 164.9303, 167.2600, 168.9342, 173.0400, 174.9670, 178.4900,
               180.9479, 183.8500, 186.2070, 190.2000, 192.2200, 195.0900, 196.9665, 200.5900,
               204.3700, 207.2000, 208.9804, 209.0000, 210.0000, 222.0000)

assert len(OCC_S) == len(OCC_P) == len(OCC_D) == MAX_Z + 1
assert len(PQN_S) == len(PQN_P) == len(PQN_D) == MAX_Z + 1
assert len(ATOM_HEAT) == len(COVALENT_RADIUS) == len(ATOMIC_MASS) == MAX_Z + 1
