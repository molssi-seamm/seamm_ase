=======
History
=======
2026.10.14.1 -- Bugfix: ASE optimizations of periodic structures had the same error
    * The ASE optimizers (used, for example, by the Structure step) built their own ASE
      Atoms with the error fixed in 2026.10.14 for the Hessian: a periodic structure
      lost its cell and its fractional coordinates were read as Cartesian. They now use
      the corrected conversion. Molecular structures are unchanged.

2026.10.14 -- Bugfix: periodic structures were passed to ASE without their cell
    * The ASE Atoms made from a periodic configuration had no cell or periodic boundary
      conditions, and their positions were the fractional coordinates read as
      Cartesian. Periodic configurations now carry their cell and Cartesian positions,
      and every configuration carries its net charge and spin multiplicity in
      ``atoms.info``. Molecular configurations are unchanged.
    * Internal: the continuous integration now installs the package and its declared
      dependencies with uv instead of a conda environment.

2025.2.7 -- Bugfix: incorrect handling of directories in the calculator
    * There was a bug in the deletion of directories on success or on error. This has
      been fixed.
      
2025.1.31 -- Improved error reporting.
    * Caught more errors and provided more informative error messages
      
2024.10.20 -- Added to the functionality and consolidated the code
    * Added method to calculate the Hessian via finite differences using ASE
    * Added references and citations where appropriate
    * Made the calculator less specific to optimization
      
2024.8.2.1 -- Calculator required to have 'implemented_properties' and 'nolabel' defined

2024.8.2 -- Initial version


