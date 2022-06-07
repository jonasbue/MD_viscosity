#!/bin/bash
dir="one_component"
d="$HOME/project/data/$dir"

for x in 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5
do
    for m in 1.0 
    do 
        for s in 1.0
        do 
            rv=$RANDOM
            rp=$RANDOM
            rt=$RANDOM

            mpirun                                          \
                -np  4 /home/christopher/lammps/build/lmp   \
                -in  in.mp_one_component        \
                -var PACKING            $x      \
                -var SEED_V             $rv     \
                -var SEED_P             $rp     \
                -var SEED_T             $rt     \
                -var DIRECTORY $dir

            echo "# rv = $rv, rp = $rp, rt = $rt" >> $d"/seeds.sh"
        done
    done
done

cp "$(readlink -f $0)" $d
cp "in.mp_one_component" $d
