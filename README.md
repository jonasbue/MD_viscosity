# project_thesis

Specialization project on Enskog-Thorne viscosity.

File structure:
|project/                   -- Main directory 
----|
    |simulation/            -- LAMMPS simulation and related files
    ----|
        |./                 -- simulation/ contains the most important scripts
        |analysis/          -- Python code to analyze LAMMPS data
        |simple_scripts/    -- Old LAMMPS scripts
        |data/              -- Where the LAMMPS output goes
----|
    report/                 -- The project report
        |./                 -- The .tex files live here
        |out/               -- Where the LaTeX output files go


To run LAMMPS:
$ lmp -in filename
$ ovito dump.lammps

TODO:
- Clean up save functions. 
  * They should save measured values along with theoretical 
    values, so the division can be done in pgfplots.
- Clean up figures
  * It should be easy to make a new for loop.
