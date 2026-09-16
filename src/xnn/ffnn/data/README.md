# Force-field parameter files (`.frc`)

Parameter libraries in the MolSSI/SEAMM `.frc` force-field format, read by
`xnn.ffnn.common.frc`. The format is documented in
`docs/how_tos/forcefield_files.rst`; in short, a file opens with a
`!MolSSI forcefield 1` line, every section starts at a `#` line and runs to
the next one, `#define` sections compose named force-field variants out of
labelled parameter sections, and `#templates` carries the SMARTS patterns that
assign atom types to a structure.

## Vendored from SEAMM (BSD-3-Clause)

`oplsaa.frc` and everything under `reaxff/` are copied verbatim from

    https://github.com/molssi-seamm/forcefield_step/tree/main/forcefield_step/data
    commit 764a20139712834a9266dc069f7abda1a4652efa (2025-05-26)

and are redistributed under the BSD-3-Clause license of that project, see
`LICENSE-SEAMM`. The scientific provenance of each parameter set is recorded
in the `#reference` sections of the files themselves.

| file | `#define` names | contents |
|---|---|---|
| `oplsaa.frc` | `oplsaa`, `CL&P`, `oplsaa+` | OPLS-AA (Jorgensen lab, with the current distribution's revisions), the CL&P ionic-liquid extension (Canongia Lopes & Padua), and their union |
| `reaxff/CHO_cho_2008.frc` | `reaxff/CHO_cho_2008` | C/H/O combustion field, Chenoweth, van Duin & Goddard, *J. Phys. Chem. A* 112, 1040 (2008) |
| `reaxff/CH_2018.frc` | `reaxff/CH_2018` | hydrocarbons (2018) |
| `reaxff/CHNO_RDX_2003.frc` | `reaxff/CHNO_RDX_2003` | nitramines, Strachan et al. (2003) |
| `reaxff/CHNO_HNS_2014.frc` | `reaxff/CHNO_HNS_2014` | HNS energetic materials (2014) |
| `reaxff/CHNOFSClNiPt_FC_2013.frc` | `reaxff/CHNOFSClNiPt_FC_2013` | fuel-cell electrocatalysis field (2013) |
| `reaxff/CHOFe_FeOH_2010.frc` | `reaxff/CHOFe_FeOH_2010` | Fe/O/H, Aryanpour et al. (2010) |
| `reaxff/CHOV_VOH_2008.frc` | `reaxff/CHOV_VOH_2008` | V/O/H, Chenoweth et al. (2008) |
| `reaxff/CHLiOFSi_Yun_2017.frc` | `reaxff/CHLiOFSi_Yun_2017` | Li-ion battery SEI (2017) |
| `reaxff/CLiSi_Guifo_2023.frc` | `reaxff/CliSi_Guifo_2023` | Li/Si anodes (2023) |
| `reaxff/HBNO_AB_2010.frc` | `reaxff/HBNO_AB_2010` | ammonia borane, Weismiller et al. (2010) |
| `reaxff/HOAu_AuO_2010.frc` | `reaxff/HOAu_AuO_2010` | Au/O/H, Joshi et al. (2010) |
| `reaxff/HOZn_ZNOH_2010.frc` | `reaxff/HOZn_ZNOH_2010` | Zn/O/H, Raymand et al. (2010) |

## Authored here

| file | `#define` names | contents |
|---|---|---|
| `lopls.frc` | `lopls` | L-OPLS, the long-hydrocarbon refit of Siu, Pluhackova & Boeckmann, *JCTC* 8, 1459 (2012), Table 2, layered over `oplsaa` via `#include` |
| `oplsaa_1996.frc` | `oplsaa-1996` | `oplsaa` with the alkane torsions of the original 1996 paper (Supporting Information Table 7), which reproduce its Table 1 |

Both compose with `oplsaa.frc` through `#include local:oplsaa.frc`; the
`local:` prefix resolves against this directory.
