#!/bin/bash

min_n=15
max_n=20
p=0.15
dropbox=../../dropbox/
data_folder=$dropbox/data/components
min_c=1
max_c=3
max_lv=4
rand=exhaust

save_fold=nodes-${min_n}-${max_n}-p-${p}-c-${min_c}-${max_c}-lv-${max_lv}
base_model_dump=$dropbox/scratch/results/graph_classification/components/$save_fold/epoch-best

output_root=./saved

if [ ! -e $output_root ];
then
    mkdir -p $output_root
fi

python er_trivial_attack.py \
    -data_folder $data_folder \
    -save_dir $output_root \
    -max_n $max_n \
    -min_n $min_n \
    -max_lv $max_lv \
    -rand_att_type $rand \
    -min_c $min_c \
    -max_c $max_c \
    -base_model_dump $base_model_dump \
    -n_graphs 5000 \
    -er_p $p \
    $@
