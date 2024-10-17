if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=104
model_name=Powerformer

#root_path_name=/pscratch/sd/k/khegazy/datasets/time_series/health/influenza/
root_path_name=/scratch/khegazy/datasets/influenza_infections/
data_path_name=national_illness.csv
model_id_name=Illness
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
        
    for pred_len in 60 #24 36 48 60
    do
        for decay_scale in 2 5 10 15 20
        do
            export CUDA_VISIBLE_DEVICES=7
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
            --n_heads 4 \
            --d_model 16 \
            --d_ff 128 \
            --dropout 0.3\
            --fc_dropout 0.3\
            --head_dropout 0\
            --patch_len 24\
            --stride 2\
            --des 'Exp' \
            --train_epochs 100\
            --lradj 'constant'\
            --itr $itr --batch_size 16 --learning_rate 0.0025\
            --attn_decay_scale ${decay_scale} \
            "$@"
            #--attn_decay_type 'zeta' \
            #>logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
        done
    done
done
