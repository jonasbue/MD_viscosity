#!/bin/bash
dir="heavy_low_exchange_rate"
d="$HOME/project/data/$dir"

for x in $(seq 0.2 0.05 0.35)
do
    for n in 0.5
    do
        for m in 10.0 50.0 
        do 
            for s in 1.0
            do 
                rv=$RANDOM
                rl=$RANDOM
                rh=$RANDOM
                rt=$RANDOM

		        mpirun                                          \
                    -np  4 /home/christopher/lammps/build/lmp   \
                    -in  ./lammps_scripts/in.mp_binary          \
                    -var PACKING            $x      \
                    -var HEAVY_FRACTION     $n      \
                    -var HEAVY_MASS         $m      \
                    -var HEAVY_DIAMETER     $s      \
                    -var SEED_V             $rv     \
                    -var SEED_L             $rl     \
                    -var SEED_H             $rh     \
                    -var SEED_T             $rt     \
                    -var DIRECTORY $d

                echo "# x = $x, n = $n, m = $m, s= $s, rv = $rv, rl = $rl, rh = $rh, rt = $rt" >> $d"/seeds.sh"
            done
        done
    done
done

cp "$(readlink -f $0)" $d
cp "in.mp_binary" $d