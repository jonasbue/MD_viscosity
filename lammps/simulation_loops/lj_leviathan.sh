#!/bin/zsh
dir="run_test"
d="$HOME/Dokumenter/skole/master/MD_viscosity/data/$dir"

for x in $(export LC_CTYPE=”en_EN.UTF-8″; seq 0.1 0.1 0.4)
do
    #x=`echo $x | sed s/,/./`
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
