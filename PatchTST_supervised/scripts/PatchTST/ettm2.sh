export CUDA_VISIBLE_DEVICES=7

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=336
model_name=PatchTST

#root_path_name=/pscratch/sd/k/khegazy/datasets/time_series/electricity/ETT-small/
root_path_name=/scratch/khegazy/datasets/electric_transformer_temperature_small/
data_path_name=ETTm2.csv
model_id_name=ETTm2
data_name=ETTm2

random_seed=2021
for pred_len in 96 192 336 720
do
    for decay_scale in 0.1 0.5 1 2 5 10
    do
        python3 -u run_longExp.py \
        --is_sequential 0 \
        --random_seed $random_seed \
        --is_training 1 \
        --root_path $root_path_name \
        --data_path $data_path_name \
        --model_id $model_id_name \
        --model $model_name \
        --data $data_name \
        --features M \
        --seq_len $seq_len \
        --pred_len $pred_len \
        --enc_in 7 \
        --e_layers 3 \
        --n_heads 16 \
        --d_model 128 \
        --d_ff 256 \
        --dropout 0.2\
        --fc_dropout 0.2\
        --head_dropout 0\
        --patch_len 16\
        --stride 8\
        --des 'Exp' \
        --train_epochs 100\
        --patience 20\
        --lradj 'TST'\
        --pct_start 0.4 \
        --itr 1 --batch_size 128 --learning_rate 0.0001 \
        --attn_decay_scale ${decay_scale} \
        "$@"
        #--attn_decay_type 'zeta' \
        #>logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
    done
done
