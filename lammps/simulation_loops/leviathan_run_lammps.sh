#!/bin/zsh
dir="run_test/"
d="$HOME/Dokumenter/skole/master/MD_viscosity/data/$dir"

for x in 0.5
do
    for n in 0.5
    do
        for m in 1.0
        do 
            for s in 1.0
            do 
                rp=$RANDOM
                rv=$RANDOM
                rt=$RANDOM

                lmp -in in.mp_lennard-jones     \
                -var PACKING            $x      \
                -var HEAVY_FRACTION     $n      \
                -var HEAVY_MASS         $m      \
                -var HEAVY_DIAMETER     $s      \
                -var SEED_P             $rp     \
                -var SEED_V             $rv     \
                -var SEED_T             $rt     \
                -var DIRECTORY $dir
                
                echo "# rp = $rp, rv = $rv, rt = $rt" >> $d"/seeds.sh"
            done
        done
    done
done

cp "$(readlink -f $0)" $d
