#large_box/varying_fraction/log.mp_N_5940_60_sigma_1_1.5_mass_1_1.0_pf_0.375.lammps
#large_box/varying_fraction/log.mp_N_3000_3000_sigma_1_1.5_mass_1_1.0_pf_0.375.lamm
#large_box/varying_fraction/log.mp_N_60_5940_sigma_1_1.5_mass_1_1.0_pf_0.475.lammps
#large_box/varying_fraction/log.mp_N_3000_3000_sigma_1_1.0_mass_1_2.0_pf_0.325.lamm
#large_box/varying_fraction/log.mp_N_5400_600_sigma_1_1.5_mass_1_1.0_pf_0.450.lammp
#large_box/varying_fraction/log.mp_N_600_5400_sigma_1_2.0_mass_1_1.0_pf_0.150.lammp
#large_box/varying_fraction/log.mp_N_3000_3000_sigma_1_1.5_mass_1_1.0_pf_0.400.lamm
#large_box/varying_fraction/log.mp_N_3000_3000_sigma_1_1.0_mass_1_1.5_pf_0.325.lamm
#large_box/varying_fraction/log.mp_N_5940_60_sigma_1_1.5_mass_1_1.0_pf_0.450.lammps
#large_box/varying_fraction/log.mp_N_5940_60_sigma_1_1.0_mass_1_1.5_pf_0.300.lammps
#large_box/varying_fraction/log.mp_N_600_5400_sigma_1_1.5_mass_1_1.0_pf_0.225.lammp

x=0.375
n=0.01
s=1.5
m=1.0
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

x=0.375
n=0.5
s=1.5
m=1.0
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
x=0.475
n=0.99
s=1.5
m=1.0
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

x=0.325
n=0.5
s=1.0
m=2.0
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

x=0.450
n=0.1
s=1.5
m=1.0
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

x=0.150
n=0.90
s=2.0
m=1.0
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

x=0.400
n=0.5
s=1.5
m=1.0
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

x=0.325
n=0.5
s=1.0
m=1.5
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

x=0.450
n=0.01
s=1.5
m=1.0
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

x=0.300
n=0.01
s=1.0
m=1.5
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

x=0.225
n=0.9
s=1.5
m=1.0
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
