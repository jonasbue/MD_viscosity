#!/bin/bash
dir="$HOME/project"
savepath="$dir/data/lj"
scriptpath="$dir/lammps_scripts"
script="in.mp_lennard-jones"

n=3000
s=1.0
# Packing fraction
for x in $(seq 0.15 0.1 0.5)
do
    # Reduced temperature
    for T in $(seq 1.3 0.1 2.0)
    do
        rp=$RANDOM  # Seed for positions
        rv=$RANDOM  # Seed for velocities
        rt=$RANDOM  # Seed for thermostat

        mpirun                                          \
            -np  8 /home/christopher/lammps/build/lmp   \
            -in  $scriptpath/$script                    \
            -var PARTICLES      $n                      \
            -var PACKING        $x                      \
            -var TEMPERATURE    $T                      \
            -var DIAMETER       $s                      \
            -var SEED_P         $rp                     \
            -var SEED_V         $rv                     \
            -var SEED_T         $rt                     \
            -var DIRECTORY      $savepath

        echo "# x = $x, T = $T, s = $s, rp = $rp, rv = $rv, rt = $rt" >> "$savepath/seeds.sh"
    done
done

cp "$(readlink -f $0)" $savepath
cp "$scriptpath/$script" $savepath

