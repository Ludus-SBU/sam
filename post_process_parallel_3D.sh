#!/bin/bash

# Written by yours truly, THE Josh Martin

n=$1
file=alltracks_3D.hdf5

echo "particle $n"

work_dir="work_${n}"
mkdir "$work_dir"
cp 400k_flash_21new.dat $work_dir
cp inlist_one_zone_burn $work_dir
cp inlist_one_zone_burn_core $work_dir
cp inlist_one_zone_burn_shell $work_dir
cp nuclides200.txt $work_dir
cp 1p1M_hybrid_to_Mch_bignet.mod $work_dir
cp one_zone_burn.data $work_dir
cp burn $work_dir
cp track_puller_3D.py $work_dir
cp make_initial_abundance_MESAhybrid_3D.py $work_dir

cd $work_dir
python3 track_puller_3D.py ../$file $n > trho_hist.dat
./burn || { echo "particle $n: burn failed" >> ../failed_particles_3D.log; 
    cd ../;
    rm -r $work_dir;
    exit 1; }
mv final_abundances.dat ../final_abundances_3D/final_abundances_$n.dat

cd ../
rm -r $work_dir