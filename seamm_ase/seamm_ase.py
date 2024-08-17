# -*- coding: utf-8 -*-

__all__ = ["SEAMM_Calculator"]

import logging

from ase.calculators.calculator import (
    Calculator as ASE_Calculator,
    all_changes as ASE_all_changes,
    register_calculator_class,
)

logger = logging.getLogger(__name__)


class SEAMM_Calculator(ASE_Calculator):
    """Generic ASE calculator for SEAMM.

    This is a generic calculator that can be used from any step in
    SEAMM to use functionality in ASE.

    The step must have a calculator method that is called by this class:

    .. code-block:: python

        def calculator(
            self,
            calculator,
            properties=["energy"],
            system_changes=ASE_all_changes,
        ):
            \"""Create a calculator for the structure step.

            Parameters
            ----------
            ase : ase.calculators.calculator.Calculator
                The ASE calculator we are working for
            properties : list of str
                The properties to calculate.
            system_changes : int
                The changes to the system.
            \"""
        ...

    An example can be found in the Structure step.

    The step must also create the SEAMM_Calculator, passing itself into the constructor,
    and set up the Atoms object to use this calculator:

    .. code-block:: python

        ...
        symbols = configuration.atoms.symbols
        XYZ = configuration.atoms.coordinates

        calculator = SEAMM_Calculator(self)
        atoms = ASE_Atoms("".join(symbols), positions=XYZ, calculator=calculator)
        ...

    The step can then call the calculate method of the SEAMM_Calculator to perform the
    calculation, or can pass the calculator to other ASE drivers that will use the
    calculator.
    """

    implemented_properties = ["energy", "forces"]
    nolabel = True

    def __init__(self, step, calculator=None, name=None, configuration=None, **kwargs):
        """
        Parameters
        ----------
        step : seamm.Node
            The step using this calculator

        **kwargs
            The keyword arguments are passed to the parent class.
        """
        self.step = step
        self.calculator = calculator  # Method or function to call
        self._name = name
        self._configuration = configuration

        super().__init__(**kwargs)

    @property
    def configuration(self):
        """The configuration this calculator represents."""
        return self._configuration

    @property
    def name(self):
        """A name for this calculator."""
        if self._name is None and self.configuration is not None:
            name = self.configuration.system.name + "/" + self.configuration.name
            return name
        else:
            return self._name

    def calculate(
        self,
        atoms=None,
        properties=["energy", "forces"],
        system_changes=ASE_all_changes,
    ):
        """Perform the calculation.

        Parameters
        ----------
        atoms : ase.Atoms
            The atoms object to calculate.
        properties : list of str
            The properties to calculate.
        system_changes : int
            The changes to the system.

        Returns
        -------
        dict
            The results of the calculation.
        """
        super().calculate(atoms, properties, system_changes)

        logger.debug(f"SEAMM_Calculator.calculate {self.name} {properties=}")
        logger.debug(f"    {system_changes=}")
        logger.debug(f"    {atoms is None=}")

        if self.calculator is None:
            self.step.calculate(self, properties, system_changes)
        else:
            self.calculator(self, properties, system_changes)

    def check_state(self, atoms, tol=1e-10):
        """Check for any system changes since last calculation."""
        return super().check_state(atoms, tol=tol)

    def get_property(self, name, atoms=None, allow_calculation=True):
        logger.debug(f"SEAMM_Calculator.get_property {self.name} {name=}")

        return super().get_property(
            name, atoms=atoms, allow_calculation=allow_calculation
        )


register_calculator_class("seamm", SEAMM_Calculator)
