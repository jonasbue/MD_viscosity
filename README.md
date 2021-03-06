# MD_viscosity
A collection of LAMMPS/Python scripts to simulate a fluid in molecular
dynamics, and compute its viscosity.

File structure:
analysis/                   -- Python code to analyze LAMMPS data
lamms/                      -- LAMMPS scripts
    |old_LAMMPS_scripts/    -- Old LAMMPS scripts
    |simulation_loops/      -- bash scripts to run LAMMPS
data/                       -- Where the LAMMPS output goes

## Analysis code overwiew

---------------------------------------------------------------
Files                           | Description
---------------------------------------------------------------
MP_from_dump_and_plotting.py    | Christopher's MP code for comparison
RDF_from_DUMP.py                | Computes RDFs from dump files
block_average.py                | Simple block averaging functions
christopher_mp.py               | Christophers MP script for comparison
convert_LAMMPS_output.py        | Converts LAMMPS output to csv format
eos.py                          | Computes the equation of state
files.py                        | Handles data files from LAMMPS
main.py                         | The same as "results", but does not save
muller_plathe.py                | The main analysis of the MP experiment
plotting.py                     | Some simple plotting functions
rdf.py                          | Saves RDF data for result plotting
rdf_christopher.py              | Christopher's script to compute RDFs
regression.py                   | Linear regression of the data
results.py                      | Functions that compute and save results
save.py                         | Functions to save data to files
tests.py                        | Test functions
utils.py                        | Utility functions
viscosity.py                    | Functions to calculate viscosities
---------------------------------------------------------------

## Next steps

- Clean up analysis code.
    - Try to make a Python module, to facilitate usage of the MP analysis code.
    - Make sure everything is well commented.
    - Restructure files to make the directory easy to navigate.
- Then, rewrite the above description.
