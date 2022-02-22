#!/bin/bash
dir="$HOME/project"
savepath="$dir/lj_data/"
scriptpath="$dir/lammps_scripts"
script="in.mp_lennard-jones"

n=3000
# Packing fraction
for x in $(seq 0.05 0.1 0.45)
do
    # Reduced temperature
    for T in 1.0 #$(seq 0.5 0.5 2.0)
    do
        # Particle diamteter
        for s in $(seq 1.0 0.5 3.0)
        do
            rp=$RANDOM  # Seed for positions
            rv=$RANDOM  # Seed for velocities
            rt=$RANDOM  # Seed for thermostat

            mpirun                                          \
                -np  6 /home/christopher/lammps/build/lmp   \
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
done

cp "$(readlink -f $0)" $savepath
cp "$scriptpath/$script" $savepath

