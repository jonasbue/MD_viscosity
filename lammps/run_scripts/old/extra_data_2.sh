#!/bin/bash
dir="data/varying_mass"
d="$HOME/project/$dir"

x=0.275
n=0.5
s=1.0
m=50.0
rv=$RANDOM
rl=$RANDOM
rh=$RANDOM
rt=$RANDOM
mpirun                              \
    -np  4 /home/christopher/lammps/build/lmp       \
    -in  in.mp_binary               \
    -var PACKING            $x      \
    -var HEAVY_FRACTION     $n      \
    -var HEAVY_MASS         $m      \
    -var HEAVY_DIAMETER     $s      \
    -var SEED_V             $rv     \
    -var SEED_L             $rl     \
    -var SEED_H             $rh     \
    -var SEED_T             $rt     \
    -var DIRECTORY $dir
echo "# rv = $rv, rl = $rl, rh = $rh, rt = $rt" >> $d"/seeds.sh"
