#!/bin/zsh
dir="$HOME/Dokumenter/skole/master/MD_viscosity"
savepath="$dir/data/run_test"
scriptpath="$dir/lammps/"
script="in.mp_lennard-jones"

for x in $(export LC_CTYPE=”en_EN.UTF-8″; seq 0.1 0.1 0.4)
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

            lmp -in             $scriptpath/$script \
            -var PACKING        $x                  \
            -var TEMPERATURE    $T                  \
            -var DIAMETER       $s                  \
            -var SEED_P         $rp                 \
            -var SEED_V         $rv                 \
            -var SEED_T         $rt                 \
            -var DIRECTORY      $savepath

            echo "# rp = $rp, rv = $rv, rt = $rt" >> $savepath"/seeds.sh"
        done
    done
done

cp "$(readlink -f $0)" $savepath
cp $scriptpath/$script $savepath
