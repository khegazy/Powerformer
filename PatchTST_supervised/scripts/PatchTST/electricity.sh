if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=336
model_name=PatchTST

root_path_name=/scratch/khegazy/datasets/electricity_consumer_load/
#root_path_name=/pscratch/sd/k/khegazy/datasets/time_series/electricity/consumer_load/
data_path_name=electricity.csv
model_id_name=Electricity
data_name=custom

random_seed=2021
for itr in 0 1 2
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
    
    for decay_scale in 0.1 #2 5 10 15 20
    do
        for pred_len in 336 #96 192 336 720
        do
            export CUDA_VISIBLE_DEVICES=2
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
            --enc_in 321 \
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
            --patience 10\
            --lradj 'TST'\
            --pct_start 0.2\
            --itr $itr --batch_size 32 --learning_rate 0.0001 \
            --attn_decay_scale ${decay_scale} \
            "$@"
            #>logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
            #--attn_decay_scale ${DECAY_SCALE} \
        done
    done
done
