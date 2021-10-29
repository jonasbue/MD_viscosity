#!/bin/zsh
for x in 0.1 0.2 0.4
do
    for n in 0.5
    do
        for m in 1.0 2.0
        do 
            for s in 1.0 2.0
            do 
                lmp -in in.mp_binary            \
                -var PACKING            $x      \
                -var HEAVY_FRACTION     $n      \
                -var HEAVY_MASS         $m      \
                -var HEAVY_DIAMETER     $s      \
                -var DIRECTORY run_test 
            done
        done
    done
done
