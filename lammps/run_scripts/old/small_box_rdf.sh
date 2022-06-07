#!/bin/bash
dir="small_box"
d="$HOME/project/$dir"

for x in 0.2 0.25 0.3
do
    for n in 0.5
    do
        for m in 50.0
        do 
            for s in 10.0
            do 
                rv=$RANDOM
                rl=$RANDOM
                rh=$RANDOM
                rt=$RANDOM

		        mpirun                                          \
                    -np  4 /home/christopher/lammps/build/lmp   \
                    -in  in.mp_binary_small_box     \
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
cp "in.small_box" $d
