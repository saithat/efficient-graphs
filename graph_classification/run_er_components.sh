#!/bin/bash

min_n=15
max_n=20
p=0.15
dropbox=../../dropbox/
data_folder=$dropbox/data/components
min_c=1
max_c=3
max_lv=5

save_fold=nodes-${min_n}-${max_n}-p-${p}-c-${min_c}-${max_c}-lv-${max_lv}
output_root=$HOME/scratch/results/graph_classification/components/$save_fold

if [ ! -e $output_root ];
then
    mkdir -p $output_root
fi

python er_components.py \
    -data_folder $data_folder \
    -save_dir $output_root \
    -max_n $max_n \
    -min_n $min_n \
    -max_lv $max_lv \
    -min_c $min_c \
    -max_c $max_c \
    -n_graphs 5000 \
    -er_p $p \
    $@
