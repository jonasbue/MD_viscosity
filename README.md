# project_thesis

Specialization project on Enskog-Thorne viscosity.

File structure:
|project/                   -- Main directory 
---\
    |simulation/            -- LAMMPS simulation and related files
    ---\
        |./                 -- simulation/ contains the most important scripts
        |analysis/          -- Python code to analyze LAMMPS data
        |simple_scripts/    -- Old LAMMPS scripts
        |data/              -- Where the LAMMPS output goes
---\
    |report/                -- The project report
        |./                 -- The .tex files live here
        |out/               -- Where the LaTeX output files go

$ lmp -in filename
$ ovito dump.lammps
