#!/bin/bash
dir="fraction_mix_data"
d="$HOME/project/$dir"

for x in $(seq 0.025 0.025 0.5)
do
    for n in 0.1 0.5 0.9
    do
        for m in 3.0
        do 
            for s in 2.0
            do 
                rv=$RANDOM
                rl=$RANDOM
                rh=$RANDOM
                rt=$RANDOM

		        mpirun                                          \
                    -np  6 /home/christopher/lammps/build/lmp   \
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

                echo "# x = $x, n = $n, m = $m, s= $s, rv = $rv, rl = $rl, rh = $rh, rt = $rt" >> $d"/seeds.sh"
            done
        done
    done
done

cp "$(readlink -f $0)" $d
cp "in.mp_binary" $d
