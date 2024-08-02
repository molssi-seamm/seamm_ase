# -*- coding: utf-8 -*-

__all__ = ["SEAMM_Calculator"]

from ase.calculators.calculator import (
    Calculator as ASE_Calculator,
    all_changes as ASE_all_changes,
)


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

    def __init__(self, step, **kwargs):
        """
        Parameters
        ----------
        step : seamm.Node
            The step using this calculator

        **kwargs
            The keyword arguments are passed to the parent class.
        """
        self.step = step
        super().__init__(**kwargs)

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

        self.step.calculator(self, properties, system_changes)
