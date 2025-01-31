# -*- coding: utf-8 -*-

__all__ = ["ASE_mixin", "SEAMM_Calculator"]

import calendar
from datetime import datetime
from importlib.resources import files
import json
import logging
from pathlib import Path
import shutil
import string
import sys
import traceback

from ase import Atoms as ASE_Atoms
from ase.calculators.calculator import (
    Calculator as ASE_Calculator,
    all_changes as ASE_all_changes,
    register_calculator_class,
)
import ase.optimize as ASE_Optimize
from ase.vibrations import Vibrations
import bibtexparser
import numpy as np
from tabulate import tabulate

from seamm_util import Q_
import seamm_util.printing as printing
from seamm_util.printing import FormattedText as __  # noqa: F401
from ._version import __version__  # noqa: F401

logger = logging.getLogger(__name__)
job = printing.getPrinter()
printer = printing.getPrinter("ASE")


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

    def __init__(
        self,
        step,
        calculator=None,
        name=None,
        configuration=None,
        **kwargs,
    ):
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
            self.step.ase_calculator(self, properties, system_changes)
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


class ASE_mixin:
    """A mix-in class for connnecting with ASE methods and calculators."""

    def ase_atoms(self, configuration):
        """Return an ASE Atoms object for the configuration.

        Parameters
        ----------
        configuration :  molsystem._Configuration
            The configuration

        Returns
        -------
        atoms : ase.Atoms()
            The ASE Atoms object.
        """
        symbols = configuration.atoms.symbols
        XYZ = configuration.atoms.coordinates

        calculator = SEAMM_Calculator(self)
        atoms = ASE_Atoms("".join(symbols), positions=XYZ, calculator=calculator)

        return atoms

    def ase_calculator(
        self,
        calculator,
        properties=["energy"],
        system_changes=ASE_all_changes,
    ):
        """Create a calculator for the structure step.

        Parameters
        ----------
        ase : ase.calculators.calculator.Calculator
            The ASE calculator we are working for
        properties : list of str
            The properties to calculate.
        system_changes : int
            The changes to the system.

        Returns
        -------
        results : dict
            The dictionary of results from the calculation.
        """
        wd = Path(self.directory)
        wd.mkdir(parents=True, exist_ok=True)

        self._step += 1
        self._results["nsteps"] = self._step
        if "step" in self._data:
            self._data["step"].append(self._step)
        fmt = "05d"

        calculator.results = {}

        n_atoms = len(calculator.atoms)
        self.logger.debug(f"{n_atoms} atoms in the structure")
        positions = calculator.atoms.positions
        self.logger.debug(f"Positions: {positions}")
        cell = calculator.atoms.cell
        self.logger.debug(f"Cell: {cell}")

        # Set the coordinates in the configuration
        self._working_configuration.atoms.set_coordinates(positions, fractionals=False)

        # Find the handler for job.out and set the level up
        job_handler = None
        out_handler = None
        for handler in job.handlers:
            if (
                isinstance(handler, logging.FileHandler)
                and "job.out" in handler.baseFilename
            ):
                job_handler = handler
                job_level = job_handler.level
                job_handler.setLevel(printing.JOB)
            elif isinstance(handler, logging.StreamHandler):
                out_handler = handler
                out_level = out_handler.level
                out_handler.setLevel(printing.JOB)

        # Get the first real node
        first_node = self.subflowchart.get_node("1").next()

        # Ensure the nodes have their options
        node = first_node
        while node is not None:
            node.all_options = self.all_options
            node = node.next()

        # And the subflowchart has the executor
        self.subflowchart.executor = self.flowchart.executor

        # Direct most output to iteration.out
        step_id = f"step_{self._step:{fmt}}"
        step_dir = Path(self.directory) / step_id
        step_dir.mkdir(parents=True, exist_ok=True)

        # A handler for the file
        if self._file_handler is not None:
            self._file_handler.close()
            job.removeHandler(self._file_handler)
        path = step_dir / "Step.out"
        path.unlink(missing_ok=True)
        self._file_handler = logging.FileHandler(path)
        self._file_handler.setLevel(printing.NORMAL)
        formatter = logging.Formatter(fmt="{message:s}", style="{")
        self._file_handler.setFormatter(formatter)
        job.addHandler(self._file_handler)

        # Add the step to the ids so the directory structure is reasonable
        self.subflowchart.reset_visited()
        self.set_subids((*self._id, step_id))

        # Run through the steps in the loop body
        node = first_node
        try:
            while node is not None:
                node = node.run()
        except DeprecationWarning as e:
            printer.normal("\nDeprecation warning: " + str(e))
            traceback.print_exc(file=sys.stderr)
            traceback.print_exc(file=sys.stdout)
        except Exception as e:
            printer.job(f"Caught exception in step {self._step}: {str(e)}")
            with open(step_dir / "stderr.out", "a") as fd:
                traceback.print_exc(file=fd)
            raise
        self.logger.debug(f"End of step {self._step}")

        # Remove any redirection of printing.
        if self._file_handler is not None:
            self._file_handler.close()
            job.removeHandler(self._file_handler)
            self._file_handler = None
        if job_handler is not None:
            job_handler.setLevel(job_level)
        if out_handler is not None:
            out_handler.setLevel(out_level)

        # Get the energy and derivatives
        paths = sorted(step_dir.glob("**/Results.json"))

        if len(paths) == 0:
            raise RuntimeError(
                "There are no energy and gradients in properties.json for step "
                f"{self._step} in {step_dir}."
            )
        else:
            # Find the most recent and assume that is the one wanted
            newest_time = None
            for path in paths:
                with path.open() as fd:
                    data = json.load(fd)
                time = datetime.fromisoformat(data["iso time"])
                if newest_time is None:
                    newest = path
                    newest_time = time
                elif time > newest_time:
                    newest_time = time
                    newest = path
            with newest.open() as fd:
                data = json.load(fd)

        energy = data["energy"]
        if "energy,units" in data:
            units = data["energy,units"]
        else:
            units = "kJ/mol"
        if "energy" in self._data:
            self._data["energy"].append(energy)

        energy *= Q_(1.0, units).to("eV").magnitude
        self._results["energy"] = Q_(energy, "eV").m_as("kJ/mol")

        gradients = data["gradients"]

        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug("\ngradients")
            for i in range(n_atoms):
                self.logger.debug(
                    f"   {gradients[i][0]:8.3f} {gradients[i][1]:8.3f} "
                    f"{gradients[i][2]:8.3f}"
                )

        if "gradients,units" in data:
            funits = data["gradients,units"]
        else:
            funits = "kJ/mol/Å"

        # Get the measures of convergence
        if "max_force" in self._data:
            max_force = np.max(np.linalg.norm(gradients, axis=1))
            self._data["max_force"].append(max_force)
            self._results["maximum_gradient"] = Q_(max_force, funits).m_as("kJ/mol/Å")
            rms_force = np.sqrt(np.mean(np.linalg.norm(gradients, axis=1) ** 2))
            self._data["rms_force"].append(rms_force)
            self._results["rms_gradient"] = Q_(rms_force, funits).m_as("kJ/mol/Å")

            if self._step > 1:
                step = positions - self._last_coordinates
                max_step = np.max(np.linalg.norm(step, axis=1))
            else:
                max_step = 0.0
            self._data["max_step"].append(max_step)
            self._results["maximum_step"] = max_step
            self._last_coordinates = np.array(positions)

        # Units!
        gradients = np.array(gradients) * Q_(1.0, funits).to("eV/Å").magnitude

        calculator.results["energy"] = energy
        calculator.results["forces"] = -gradients

        # Log the results
        if self._logfile is not None:
            headers = [
                "Step",
                f"E ({units})",
                f"Fmax ({funits})",
                f"Frms ({funits})",
                "max step (Å)",
            ]
            tmp = tabulate(
                self._data,
                headers=headers,
                tablefmt="rounded_outline",
                disable_numparse=False,
                floatfmt=".3f",
            )
            with open(self._logfile, "w") as fd:
                fd.write(tmp)
                fd.write("\n")

        # and plot the results
        if "step" in self._data:
            self.plot(E_units=units, F_units=funits)

        # Citation!
        self.ase_read_bibliography()
        if "seamm_ase" in self._bibliography:
            self.references.cite(
                raw=self._bibliography["seamm_ase"],
                alias="seamm_ase",
                module="seamm_ase",
                level=self.citation_level,
                note=("The principle citation for the ASE connector for SEAMM."),
            )

    def ase_read_bibliography(self):
        """Read the bibliography from a file and add to the local bibliography"""
        self.logger.debug("Reading the seamm_ase bibliography")
        if "seamm-ase" not in self._bibliography:
            try:
                data = files("seamm_ase.data").joinpath("references.bib").read_text()
                tmp = bibtexparser.loads(data).entries_dict
                writer = bibtexparser.bwriter.BibTexWriter()
                for key, data in tmp.items():
                    self.logger.debug(f"      {key}")
                    self._bibliography[key] = writer._entry_to_bibtex(data)
            except Exception as e:
                self.logger.info(f"Exception in reading seamm_ase bibliography: {e}")
                pass
        if "seamm_ase" in self._bibliography:
            try:
                template = string.Template(self._bibliography["seamm_ase"])

                if "untagged" in __version__ or "unknown" in __version__:
                    # Development version
                    year = datetime.now().year
                    month = datetime.now().month
                else:
                    year, month = __version__.split(".")[0:2]
                try:
                    month = calendar.month_abbr[int(month)].lower()
                except Exception:
                    year = datetime.now().year
                    month = datetime.now().month
                    month = calendar.month_abbr[int(month)].lower()

                citation = template.substitute(
                    month=month, version=__version__, year=str(year)
                )

                self._bibliography["seamm_ase"] = citation
            except Exception as e:
                printer.important(f"Exception in citation {type(e)}: {e}")
                printer.important(traceback.format_exc())

    def run_ase_Hessian(
        self,
        step_size=0.01,
        on_error="keep last subdirectory",
        on_success="delete all subdirectories",
    ):
        """Run ASE to get the Hessian

        Parameters
        ----------
        step_size : float
            The finite-difference step, in Angstrom

        Returns
        -------
        vibrations : ASE VibrationsData() object
        """
        # Citation!
        self.ase_read_bibliography()
        if "ASE" in self._bibliography:
            self.references.cite(
                raw=self._bibliography["ASE"],
                alias="ASE",
                module="seamm_ase",
                level=1,
                note="Main reference for ASE.",
            )

        self._last_coordinates = None
        self._step = 0

        _, starting_configuration = self.get_system_configuration()
        wd = Path(self.directory)
        wd.mkdir(parents=True, exist_ok=True)

        # Do not log the energies, etc.
        self._logfile = None

        atoms = self.ase_atoms(starting_configuration)

        # The ASE Vibrations object
        vibrations = Vibrations(atoms, name=wd / "finite-difference", delta=step_size)

        # Run the finite-difference calculation of the Hessian
        caught_error = False
        exception = None
        try:
            vibrations.run()
            result = vibrations.get_vibrations()
        except Exception as e:  # noqa: F841
            exception = e
            caught_error = True
            text = "".join(traceback.format_exception(e))
            printer.important(
                f"Exception in the finite-difference calculation!\n\n{text}"
            )

        # Clean up the subdirectories
        if caught_error:
            if "all" in on_error:
                subdirectories = wd.glob("step_*")
                for subdirectory in subdirectories:
                    shutil.rmtree(subdirectory)
            elif "last" in on_error:
                subdirectories = wd.glob("step_*")
                subdirectories = sorted(subdirectories)
                for subdirectory in subdirectories[:-1]:
                    shutil.rmtree(subdirectory)
            raise exception from None
        else:
            if "all" in on_success:
                subdirectories = wd.glob("step_*")
                for subdirectory in subdirectories:
                    shutil.rmtree(subdirectory)
            elif "last" in on_success:
                subdirectories = wd.glob("step_*")
                subdirectories = sorted(subdirectories)
                for subdirectory in subdirectories[:-1]:
                    shutil.rmtree(subdirectory)

        return result

    def run_ase_optimizer(self, P, PP):
        """Run a Structure step.

        Parameters
        ----------
        P : dict
            The current values of the parameters
        PP : dict
            The current values of the parameters, formatted for printing
        """
        # Citation!
        self.ase_read_bibliography()
        if "ASE" in self._bibliography:
            self.references.cite(
                raw=self._bibliography["ASE"],
                alias="ASE",
                module="seamm_ase",
                level=1,
                note="Main reference for ASE.",
            )

        self._data = {
            "step": [],
            "energy": [],
            "max_force": [],
            "rms_force": [],
            "max_step": [],
        }
        self._last_coordinates = None
        self._step = 0

        _, starting_configuration = self.get_system_configuration()
        n_atoms = starting_configuration.n_atoms

        # Print what we are doing
        printer.important(
            __(self.description_text(P, short=True, natoms=n_atoms), indent=self.indent)
        )

        # Create the directory
        wd = Path(self.directory)
        wd.mkdir(parents=True, exist_ok=True)

        # Setup the log file for the optimization
        self._logfile = wd / "optimization.log"

        symbols = starting_configuration.atoms.symbols
        XYZ = starting_configuration.atoms.coordinates

        calculator = SEAMM_Calculator(self)
        atoms = ASE_Atoms("".join(symbols), positions=XYZ, calculator=calculator)

        # The default maximum number of steps may vary depending on the optimizer
        max_steps = P["max steps"]

        # Optimize the structure
        optimizer = P["optimizer"].split("/")[0].lower()
        if optimizer == "bfgs":
            optimizer = ASE_Optimize.BFGS(atoms, restart=wd / "bfgs.json", logfile=None)
        elif optimizer == "lbfgs":
            optimizer = ASE_Optimize.LBFGS(
                atoms, restart=wd / "lbfgs.json", logfile=None
            )
        elif optimizer == "fire":
            optimizer = ASE_Optimize.FIRE(atoms, restart=wd / "fire.json", logfile=None)
        elif optimizer == "gpmin":
            optimizer = ASE_Optimize.GPMin(
                atoms, restart=wd / "gpmin.json", logfile=None
            )
        elif optimizer == "mdmin":
            optimizer = ASE_Optimize.MDMin(
                atoms, restart=wd / "mdmin.json", logfile=None
            )
        elif optimizer == "bfgslinesearch":
            optimizer = ASE_Optimize.BFGSLineSearch(
                atoms, restart=wd / "bfgsline.json", logfile=None
            )
        elif optimizer == "lbfgslinesearch":
            optimizer = ASE_Optimize.LBFGSLineSearch(
                atoms, restart=wd / "lbfgsline.json", logfile=None
            )
        else:
            raise ValueError(
                f"Unknown optimizer '{optimizer}' ({P['optimizer']}) in Structure"
            )

        convergence = P["Maximum atomic gradient criterion"].m_as("eV/Å")
        if "natoms" in max_steps:
            tmp = max_steps.split()
            if "natoms" in tmp[0]:
                max_steps = int(tmp[1]) * len(atoms)
            else:
                max_steps = int(tmp[0]) * len(atoms)

        # Run the optimization
        exception = None
        try:
            converged = optimizer.run(fmax=convergence, steps=max_steps)
        except Exception as exception:  # noqa: F841
            print(f"Exception: {exception}")
            converged = False
        finally:
            self._results["converged"] = converged

        # Clean up the subdirectories
        if exception is not None or not converged:
            keep = P["on error"]
            if keep == "delete all subdirectories":
                subdirectories = wd.glob("step_*")
                for subdirectory in subdirectories:
                    shutil.rmtree(subdirectory)
            elif keep == "keep last subdirectory":
                subdirectories = wd.glob("step_*")
                subdirectories = sorted(subdirectories)
                for subdirectory in subdirectories[:-1]:
                    shutil.rmtree(subdirectory)
            if not converged:
                raise RuntimeError(
                    f"Optimization did not converge in {max_steps} steps"
                )
            raise exception from None
        else:
            keep = P["on success"]
            if keep == "delete all subdirectories":
                subdirectories = wd.glob("step_*")
                for subdirectory in subdirectories:
                    shutil.rmtree(subdirectory)
            elif keep == "keep last subdirectory":
                subdirectories = wd.glob("step_*")
                subdirectories = sorted(subdirectories)
                for subdirectory in subdirectories[:-1]:
                    shutil.rmtree(subdirectory)
