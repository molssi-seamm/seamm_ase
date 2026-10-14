"""
Unit and regression test for the seamm_ase package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest  # noqa: F401

import seamm_ase  # noqa: F401


def test_seamm_ase_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "seamm_ase" in sys.modules


import numpy as np  # noqa: E402

from seamm_ase import ASE_mixin, SEAMM_Calculator  # noqa: E402


class _Atoms:
    """A molsystem-like atoms table: fractional coordinates for a periodic cell."""

    def __init__(self, symbols, fractional, cell):
        self.symbols = symbols
        self._frac = np.asarray(fractional, dtype=float)
        self._cell = cell

    def get_coordinates(self, fractionals=True):
        if fractionals and self._cell is not None:
            return self._frac.tolist()
        if self._cell is None:
            return self._frac.tolist()
        return (self._frac @ np.asarray(self._cell)).tolist()

    @property
    def coordinates(self):
        return self.get_coordinates()


class _Cell:
    def __init__(self, vectors):
        self._vectors = vectors

    def vectors(self):
        return self._vectors


class _Configuration:
    def __init__(self, symbols, xyz, cell=None, charge=0, multiplicity=1):
        self.atoms = _Atoms(symbols, xyz, cell)
        self.cell = _Cell(cell)
        self.periodicity = 0 if cell is None else 3
        self.charge = charge
        self.spin_multiplicity = multiplicity


class _Step(ASE_mixin):
    pass


def test_periodic_configuration_has_cell_and_cartesian_positions():
    cell = [[10.0, 0.0, 0.0], [0.0, 11.0, 0.0], [1.0, 0.0, 12.0]]
    frac = [[0.1, 0.2, 0.3], [0.5, 0.5, 0.5]]
    config = _Configuration(["Li", "F"], frac, cell=cell, charge=0)
    atoms = _Step().ase_atoms(config)
    assert atoms.pbc.all()
    assert np.allclose(atoms.cell[:], cell)
    assert np.allclose(atoms.positions, np.asarray(frac) @ np.asarray(cell))
    assert atoms.get_chemical_symbols() == ["Li", "F"]
    assert isinstance(atoms.calc, SEAMM_Calculator)


def test_molecular_configuration_charge_and_spin():
    xyz = [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0]]
    config = _Configuration(["O", "H"], xyz, charge=-1, multiplicity=1)
    atoms = _Step().ase_atoms(config)
    assert not atoms.pbc.any()
    assert np.allclose(atoms.positions, xyz)
    assert atoms.info["charge"] == -1 and atoms.info["spin_multiplicity"] == 1


def test_symbols_are_not_reparsed():
    # "C" + "O" must not become cobalt, nor "S" + "N" tin
    config = _Configuration(
        ["C", "O", "S", "N"], [[0, 0, 0], [1.1, 0, 0], [3, 0, 0], [4.5, 0, 0]]
    )
    assert _Step().ase_atoms(config).get_chemical_symbols() == ["C", "O", "S", "N"]


def test_with_a_molsystem_configuration():
    molsystem = pytest.importorskip("molsystem")
    db = molsystem.SystemDB(filename="file:seamm_ase_test?mode=memory&cache=shared")
    try:
        config = db.create_system(name="s").create_configuration(name="c")
        config.periodicity = 3
        config.coordinate_system = "fractional"
        config.cell.parameters = [8.0, 9.0, 10.0, 90.0, 90.0, 90.0]
        config.atoms.append(
            x=[0.25, 0.75], y=[0.5, 0.5], z=[0.1, 0.9], symbol=["Na", "Cl"]
        )
        atoms = _Step().ase_atoms(config)
        assert np.allclose(atoms.cell.lengths(), [8.0, 9.0, 10.0]) and atoms.pbc.all()
        assert np.allclose(atoms.positions[0], [2.0, 4.5, 1.0])
        assert np.allclose(atoms.positions[1], [6.0, 4.5, 9.0])
    finally:
        db.close()
