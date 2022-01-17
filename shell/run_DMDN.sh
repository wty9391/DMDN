#!/usr/bin/env bash

# advertiser-bid_scale
#declare -a dictionaries=("1458 0.2" "2259 0.5" "2261 0.2" "2821 0.7" "2997 0.6" "3358 0.2" "3386 1.0" "3427 1.0" "3476 1.5")
declare -a dictionaries=("2821 0.7" "2997 0.6")

dataset_root="./datasets/make-ipinyou-data"
CUDA_VISIBLE_DEVICES="0"
batch_size=4096
hidden_dimension=512
epoch=10
learning_rate=0.00001
K=10
price_scale=1.0
debug=1
overwrite_output_dir="False"
family="gaussian"
dataset="ipinyou"
seed=9391
drop_out=0.2
first_price="False"
nomalize_loss="False"

cd ..

for dic in "${dictionaries[@]}"; do
    read -a arr <<< "$dic"  # uses default whitespace IFS
    mkdir -p ./result/${arr[0]}/log/run_DMDN
    echo "python ./DMDN.py --dataset_root $dataset_root \
    --dataset $dataset \
    --advertiser ${arr[0]} \
    --output_dir ./result/${arr[0]} \
    --overwrite_output_dir $overwrite_output_dir \
    --CUDA_VISIBLE_DEVICES $CUDA_VISIBLE_DEVICES \
    --batch_size $batch_size \
    --hidden_dimension $hidden_dimension \
    --epoch $epoch \
    --learning_rate $learning_rate \
    --K $K \
    --price_scale $price_scale \
    --bid_scale ${arr[1]} \
    --debug $debug \
    --family $family \
    --seed $seed \
    --drop_out $drop_out \
    --first_price $first_price\
    --nomalize_loss $nomalize_loss"

    python ./DMDN.py --dataset_root $dataset_root \
    --dataset $dataset \
    --advertiser ${arr[0]} \
    --output_dir ./result/${arr[0]} \
    --overwrite_output_dir $overwrite_output_dir \
    --CUDA_VISIBLE_DEVICES $CUDA_VISIBLE_DEVICES \
    --batch_size $batch_size \
    --hidden_dimension $hidden_dimension \
    --epoch $epoch \
    --learning_rate $learning_rate \
    --K $K \
    --price_scale $price_scale \
    --bid_scale ${arr[1]} \
    --debug $debug \
    --family $family \
    --seed $seed \
    --drop_out $drop_out \
    --first_price $first_price \
    --nomalize_loss $nomalize_loss \
        1>"./result/${arr[0]}/log/run_DMDN/1.log" 2>"./result/${arr[0]}/log/run_DMDN/2.log"&
done