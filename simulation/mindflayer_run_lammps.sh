#!/bin/bash
dir="run_test"
#d="$HOME/Dokumenter/project/simulation/data/$dir"
for x in 0.1 0.2 0.3
do
    for n in 0.5
    do
        for m in 1.0 1.5 2.0
        do 
            for s in 1.0
            do 
                rv=$RANDOM
                rl=$RANDOM
                rh=$RANDOM
                rt=$RANDOM

		        mpirun                              \
                    -np  4 ~/lammps/build/lmp       \
                    -in  in.mp_binary               \
                    -var PACKING            $x      \
                    -var HEAVY_FRACTION     $n      \
                    -var HEAVY_MASS         $m      \
                    -var HEAVY_DIAMETER     $s      \
                    -var SEED_V             $rv     \
                    -var SEED_L             $rl     \
                    -var SEED_H             $rh     \
                    -var SEED_T             $rt     \
                    -var DIRECTORY $dir

                echo "# rv = $rv, rl = $rl, rh = $rh, rt = $rt" >> $d"/seeds.sh"
            done
        done
    done
done

cp "$(readlink -f $0)" $d
