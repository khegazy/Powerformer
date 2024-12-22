# ALL scripts in this file come from Autoformer
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

random_seed=2021
model_name=Transformer
root_path_name=/scratch/khegazy/datasets/electric_transformer_temperature_small/
data_path_name=ETTh2.csv
model_id_name=ETTh2
data_name=ETTh2

for itr in 0 1 2
do
    for seq_len in 256
    do
        if [ "$itr" -eq "0" ]
        then
            random_seed=2021
        elif [ "$itr" -eq "1" ]
        then
            random_seed=1776
        elif [ "$itr" -eq "2" ]
        then
            random_seed=1953
        else
            exit
        fi

        for decay_scale in 0.1 0.25 0.5 0.75 1
        do
            for pred_len in 96 192 336 720
            do
            export CUDA_VISIBLE_DEVICES=1
            python3 -u run_longExp.py \
                --random_seed $random_seed \
                --is_training 1 \
                --root_path $root_path_name \
                --data_path $data_path_name \
                --model_id $model_id_name \
                --model $model_name \
                --data $data_name \
                --features M \
                --seq_len $seq_len \
                --label_len 48 \
                --pred_len $pred_len \
                --e_layers 2 \
                --d_layers 1 \
                --factor 3 \
                --enc_in 7 \
                --dec_in 7 \
                --c_out 7 \
                --des 'Exp' \
                --itr $itr \
                --attn_decay_scale ${decay_scale} \
                "$@" #>logs/LongForecasting/$model_name'_Etth2_'$pred_len.log
            done
        done
    done
done