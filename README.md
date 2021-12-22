# project_thesis

Specialization project on Enskog-Thorne viscosity.

File structure:
|simulation/            -- LAMMPS simulation and related files
----|
    |./                 -- simulation/ contains the most important scripts
    |analysis/          -- Python code to analyze LAMMPS data
    |simple_scripts/    -- Old LAMMPS scripts
    |data/              -- Where the LAMMPS output goes

In repo "project_report":
----|
    report/                 -- The project report
        |./                 -- The .tex files live here
        |out/               -- Where the LaTeX output files go


To run LAMMPS:
$ lmp -in filename
$ ovito dump.lammps

TODO:
- RDF at two different times
- RDF at contact for large sigma
- Correct results text
- Correct discussion
- Write inroduction
- Write method
