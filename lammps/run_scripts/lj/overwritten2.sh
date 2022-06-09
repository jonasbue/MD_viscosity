#!/bin/bash
dir="$HOME/project"
savepath="$dir/data/lj"
scriptpath="$dir/lammps_scripts"
script="in.mp_lennard-jones"

n=3000
s=1.0
# Packing fraction
for x in 0.15 0.25 0.35 0.4
do
    # Reduced temperature
    for T in 2.1 2.3 2.4 2.5
    do
        logname="${savepath}/log.mp_N_${n}_sigma_1_temp_${T}_pf_${x}.lammps"
        echo $logname
        if [[ ! -f $logname ]]
        then
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
        fi
    done
done

cp "$(readlink -f $0)" $savepath
cp "$scriptpath/$script" $savepath

