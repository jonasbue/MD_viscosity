#!/bin/bash
dir="$HOME/project"
savepath="$dir/test/"
scriptpath="$dir/lammps_scripts"
script="in.mp_lennard-jones"

n=100
s=1.0
# Packing fraction
for x in $(seq 0.3 0.1 0.35)
do
    # Reduced temperature
    for T in $(seq 1.0 0.5 1.1)
    do
        rp=$RANDOM  # Seed for positions
        rv=$RANDOM  # Seed for velocities
        rt=$RANDOM  # Seed for thermostat

        mpirun                                          \
            -np  1 /home/christopher/lammps/build/lmp   \
            -in  $scriptpath/$script                    \
            -var PARTICLES      $n                      \
            -var PACKING        $x                      \
            -var TEMPERATURE    $T                      \
            -var DIAMETER       $s                      \
            -var SEED_P         $rp                     \
            -var SEED_V         $rv                     \
            -var SEED_T         $rt                     \
            -var DIRECTORY      $savepath

        echo "# x = $x, T = $T, rp = $rp, rv = $rv, rt = $rt" >> "$savepath/seeds.sh"
    done
done

cp "$(readlink -f $0)" $savepath
cp "$scriptpath/$script" $savepath

