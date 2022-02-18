#!/bin/bash
dir="$HOME/project/data"
savepath="$dir/run_test"
scriptpath="$dir"
script="in.mp_lennard-jones"

# Packing fraction
for x in 0.1 
do
    # Reduced temperature
    for T in 1.5
    do
        # Particle diamteter
        for s in 1.5
        do
            rp=$RANDOM  # Seed for positions
            rv=$RANDOM  # Seed for velocities
            rt=$RANDOM  # Seed for thermostat

            mpirun                                          \
                -np  4 /home/christopher/lammps/build/lmp   \
                -in  $scriptpath/$script                    \
                -var PACKING        $x                      \
                -var TEMPERATURE    $T                      \
                -var DIAMETER       $s                      \
                -var SEED_P         $rp                     \
                -var SEED_V         $rv                     \
                -var SEED_T         $rt                     \
                -var DIRECTORY      $savepath

        echo "# x = $x, T = $T, s = $s, rv = $rv, rl = $rl, rh = $rh, rt = $rt" >> $d"/seeds.sh"
    done
done

cp "$(readlink -f $0)" $savepath
cp $scriptpath/$script $savepath
