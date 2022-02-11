#!/bin/zsh
dir="run_test"
d="$HOME/Dokumenter/skole/master/MD_viscosity/data/$dir"

for ((x=0.1; x<=0.4; x+=0.1))
do
    for n in 1000
    do
        for m in 1.0
        do 
            for c in 1.0
            do 
                rp=$RANDOM
                rv=$RANDOM
                rt=$RANDOM

                lmp -in lammps/in.mp_lennard-jones  \
                -var PACKING            $x          \
                -var SEED_P             $rp         \
                -var SEED_V             $rv         \
                -var SEED_T             $rt         \
                -var CUT                $c          \
                -var DIRECTORY          $d
                
                echo "# rp = $rp, rv = $rv, rt = $rt" >> $d"/seeds.sh"
            done
        done
    done
done

cp "$(readlink -f $0)" $d
