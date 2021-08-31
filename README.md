# project_thesis

Specialization project on Enskog-Thorne viscosity.

## LAMMPS workflow

To run and animate LAMMPS scripts, use the following workflow

$ lmp -in filename
$ convert -delay 8 out/imagename.*.jpg ./gifname.gif && eog ./gifname.gif
$ gifsicle --optimize gifname.gif --colors 256 -o optimizedgifname.gif

This should work, provided that an image dump has been made in LAMMPS.
Imagemagick fails (and/or runs slowly) for a large number of images,
so if a large animation is really needed, then try ffmpeg etc. instead.
