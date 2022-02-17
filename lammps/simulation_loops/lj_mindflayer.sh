#!/bin/bash
dir="run_test"
d="$HOME/project/data/$dir"

for x in 0.1 
do
    for T in 1.5
    do
        rv=$RANDOM
        rl=$RANDOM
        rh=$RANDOM
        rt=$RANDOM

        mpirun                                          \
            -np  4 /home/christopher/lammps/build/lmp   \
            -in  in.mp_binary               \
            -var PACKING            $x      \
            -var TEMPERATURE        $T      \
            -var SEED_V             $rv     \
            -var SEED_L             $rl     \
            -var SEED_H             $rh     \
            -var SEED_T             $rt     \
            -var DIRECTORY $dir

        echo "# x = $x, T = $T, rv = $rv, rl = $rl, rh = $rh, rt = $rt" >> $d"/seeds.sh"
    done
done

cp "$(readlink -f $0)" $d
cp "in.mp_binary" $d
