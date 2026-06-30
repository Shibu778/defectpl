# -*- coding: utf-8 -*-
"""
Tests for defectpl.io.wavecar — VASP file I/O helpers.

All tests use synthetic in-memory data so no real VASP files are needed.
"""

from __future__ import annotations

import gzip
from textwrap import dedent

import numpy as np
import pytest

from defectpl.io.wavecar import (
    open_text,
    read_ibzkpt_weights,
    read_oszicar,
    read_outcar_energy,
    read_outcar_fermi,
    read_poscar,
    get_total_energy,
    get_fermi_level,
    get_structure,
    _ATOMIC_MASS,
)


# ===========================================================================
# Fixtures: synthetic VASP files in a temp directory
# ===========================================================================

OSZICAR_CONTENT = dedent("""\
    N       E                     dE             d eps       ncg     rms          rms(c)
          1 F= -.54321000E+02 E0= -.54321000E+02  d E =-.54321000E+02
          2 F= -.54322000E+02 E0= -.54322100E+02  d E =-.10000000E-02
""")

IBZKPT_CONTENT = dedent("""\
    Automatically generated mesh
             4
    Reciprocal lattice
      0.00000000  0.00000000  0.00000000       4
      0.50000000  0.00000000  0.00000000       3
      0.00000000  0.50000000  0.00000000       3
      0.50000000  0.50000000  0.00000000       2
""")

OUTCAR_ENERGY_CONTENT = dedent("""\
    ICHARG =  2    charge: 0
     ik=  1  ...
      energy(sigma->0)  =    -54.31000000
      energy(sigma->0)  =    -54.32100000
""")

OUTCAR_FERMI_CONTENT = dedent("""\
    ICHARG =  2
     E-fermi :   5.1234     XC(G=0): -12.345  alpha+bet : -0.0100
""")

POSCAR_DIRECT = dedent("""\
    Diamond cubic Si
       1.0
         3.86746  0.00000  0.00000
         0.00000  3.86746  0.00000
         0.00000  0.00000  3.86746
    Si
    2
    Direct
      0.00000  0.00000  0.00000
      0.25000  0.25000  0.25000
""")

POSCAR_CART = dedent("""\
    NV center
       1.0
         5.00000  0.00000  0.00000
         0.00000  5.00000  0.00000
         0.00000  0.00000  5.00000
    C   N
    1   1
    Cartesian
      0.00000  0.00000  0.00000
      1.25000  1.25000  1.25000
""")


@pytest.fixture()
def tmpdir(tmp_path):
    return tmp_path


@pytest.fixture()
def oszicar_file(tmpdir):
    p = tmpdir / "OSZICAR"
    p.write_text(OSZICAR_CONTENT, encoding="utf-8")
    return p


@pytest.fixture()
def oszicar_gz(tmpdir):
    p = tmpdir / "OSZICAR.gz"
    with gzip.open(p, "wt", encoding="utf-8") as fh:
        fh.write(OSZICAR_CONTENT)
    return p


@pytest.fixture()
def ibzkpt_file(tmpdir):
    p = tmpdir / "IBZKPT"
    p.write_text(IBZKPT_CONTENT, encoding="utf-8")
    return p


@pytest.fixture()
def outcar_file(tmpdir):
    p = tmpdir / "OUTCAR"
    content = OUTCAR_ENERGY_CONTENT + OUTCAR_FERMI_CONTENT
    p.write_text(content, encoding="utf-8")
    return p


@pytest.fixture()
def poscar_direct(tmpdir):
    p = tmpdir / "POSCAR"
    p.write_text(POSCAR_DIRECT, encoding="utf-8")
    return p


@pytest.fixture()
def poscar_cart(tmpdir):
    p = tmpdir / "POSCAR_cart"
    p.write_text(POSCAR_CART, encoding="utf-8")
    return p


# ===========================================================================
# open_text
# ===========================================================================


class TestOpenText:
    def test_plain_text(self, tmpdir):
        p = tmpdir / "test.txt"
        p.write_text("hello\nworld\n", encoding="utf-8")
        with open_text(p) as fh:
            content = fh.read()
        assert "hello" in content

    def test_gz_text(self, tmpdir):
        p = tmpdir / "test.txt.gz"
        with gzip.open(p, "wt", encoding="utf-8") as fh:
            fh.write("hello compressed\n")
        with open_text(p) as fh:
            content = fh.read()
        assert "hello compressed" in content


# ===========================================================================
# read_ibzkpt_weights
# ===========================================================================


class TestReadIbzkptWeights:
    def test_correct_weights(self, ibzkpt_file):
        w = read_ibzkpt_weights(ibzkpt_file)
        np.testing.assert_array_equal(w, [4, 3, 3, 2])

    def test_length(self, ibzkpt_file):
        w = read_ibzkpt_weights(ibzkpt_file)
        assert len(w) == 4

    def test_sum_normalisation(self, ibzkpt_file):
        w = read_ibzkpt_weights(ibzkpt_file)
        np.testing.assert_almost_equal(w.sum(), 12.0)

    def test_gz_ibzkpt(self, tmpdir):
        p = tmpdir / "IBZKPT.gz"
        with gzip.open(p, "wt") as fh:
            fh.write(IBZKPT_CONTENT)
        w = read_ibzkpt_weights(p)
        assert w[0] == 4


# ===========================================================================
# read_oszicar
# ===========================================================================


class TestReadOszicar:
    def test_final_energy(self, oszicar_file):
        data = read_oszicar(oszicar_file)
        assert pytest.approx(data["final_energy"], rel=1e-6) == -54.3221

    def test_free_energy(self, oszicar_file):
        data = read_oszicar(oszicar_file)
        assert pytest.approx(data["free_energy"], rel=1e-6) == -54.3220

    def test_n_steps(self, oszicar_file):
        data = read_oszicar(oszicar_file)
        assert len(data["steps"]) == 2

    def test_gz_oszicar(self, oszicar_gz):
        data = read_oszicar(oszicar_gz)
        assert data["final_energy"] < 0.0

    def test_missing_steps_raises(self, tmpdir):
        p = tmpdir / "OSZICAR_empty"
        p.write_text("no ionic steps here\n", encoding="utf-8")
        with pytest.raises(ValueError, match="No ionic steps"):
            read_oszicar(p)


# ===========================================================================
# read_outcar_energy / fermi
# ===========================================================================


class TestReadOutcar:
    def test_energy(self, outcar_file):
        E = read_outcar_energy(outcar_file)
        assert pytest.approx(E, rel=1e-5) == -54.321

    def test_energy_last_occurrence(self, outcar_file):
        E = read_outcar_energy(outcar_file)
        assert pytest.approx(E, abs=1e-4) == -54.321

    def test_fermi(self, outcar_file):
        ef = read_outcar_fermi(outcar_file)
        assert pytest.approx(ef, abs=1e-4) == 5.1234

    def test_missing_energy_raises(self, tmpdir):
        p = tmpdir / "OUTCAR"
        p.write_text("nothing here\n", encoding="utf-8")
        with pytest.raises(ValueError, match="energy"):
            read_outcar_energy(p)

    def test_missing_fermi_raises(self, tmpdir):
        p = tmpdir / "OUTCAR"
        p.write_text("nothing here\n", encoding="utf-8")
        with pytest.raises(ValueError, match="E-fermi"):
            read_outcar_fermi(p)


# ===========================================================================
# read_poscar
# ===========================================================================


class TestReadPoscar:
    def test_direct_coordinates_shape(self, poscar_direct):
        s = read_poscar(poscar_direct)
        assert s["positions"].shape == (2, 3)

    def test_direct_coordinates_first_atom(self, poscar_direct):
        s = read_poscar(poscar_direct)
        np.testing.assert_array_almost_equal(s["positions"][0], [0, 0, 0])

    def test_natoms(self, poscar_direct):
        s = read_poscar(poscar_direct)
        assert s["natoms"] == 2

    def test_species(self, poscar_direct):
        s = read_poscar(poscar_direct)
        assert s["species"] == ["Si"]

    def test_atom_species_repeated(self, poscar_direct):
        s = read_poscar(poscar_direct)
        assert s["atom_species"] == ["Si", "Si"]

    def test_lattice_shape(self, poscar_direct):
        s = read_poscar(poscar_direct)
        assert s["lattice"].shape == (3, 3)

    def test_cartesian_poscar(self, poscar_cart):
        s = read_poscar(poscar_cart)
        assert s["natoms"] == 2
        # First atom should be at fractional [0, 0, 0]
        np.testing.assert_array_almost_equal(s["positions"][0], [0, 0, 0], decimal=5)

    def test_masses_lookup(self, poscar_direct):
        s = read_poscar(poscar_direct)
        for m in s["masses"]:
            assert m > 0.0

    def test_atomic_mass_dict_coverage(self):
        assert "C" in _ATOMIC_MASS
        assert "N" in _ATOMIC_MASS
        assert "Si" in _ATOMIC_MASS
        assert "Fe" in _ATOMIC_MASS


# ===========================================================================
# get_total_energy / get_fermi_level / get_structure
# ===========================================================================


class TestAutoDetect:
    def test_get_total_energy_oszicar(self, tmpdir):
        (tmpdir / "OSZICAR").write_text(OSZICAR_CONTENT, encoding="utf-8")
        E = get_total_energy(tmpdir, prefer="oszicar")
        assert E < 0.0

    def test_get_total_energy_outcar_fallback(self, tmpdir):
        (tmpdir / "OUTCAR").write_text(
            OUTCAR_ENERGY_CONTENT + OUTCAR_FERMI_CONTENT, encoding="utf-8"
        )
        E = get_total_energy(tmpdir, prefer="outcar")
        assert E < 0.0

    def test_get_total_energy_no_files(self, tmpdir):
        with pytest.raises(FileNotFoundError):
            get_total_energy(tmpdir)

    def test_get_fermi_level(self, tmpdir):
        (tmpdir / "OUTCAR").write_text(
            OUTCAR_ENERGY_CONTENT + OUTCAR_FERMI_CONTENT, encoding="utf-8"
        )
        ef = get_fermi_level(tmpdir)
        assert pytest.approx(ef, abs=1e-4) == 5.1234

    def test_get_structure_poscar(self, tmpdir):
        (tmpdir / "POSCAR").write_text(POSCAR_DIRECT, encoding="utf-8")
        s = get_structure(tmpdir, relaxed=False)
        assert s["natoms"] == 2

    def test_get_structure_prefers_contcar(self, tmpdir):
        (tmpdir / "POSCAR").write_text(POSCAR_DIRECT, encoding="utf-8")
        contcar_content = POSCAR_DIRECT.replace("Diamond cubic Si", "CONTCAR title")
        (tmpdir / "CONTCAR").write_text(contcar_content, encoding="utf-8")
        s = get_structure(tmpdir, relaxed=True)
        assert "CONTCAR" in s["title"]

    def test_get_structure_not_found(self, tmpdir):
        with pytest.raises(FileNotFoundError):
            get_structure(tmpdir)
