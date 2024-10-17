
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=96
model_name=Powerformer

#root_path_name=/pscratch/sd/k/khegazy/datasets/time_series/weather/US_hourly_1.6K-stations/
root_path_name=/scratch/khegazy/datasets/inverse_PL/
data_path_name=neg_alpha_noise-0.25.npy
model_id_name=InversePL0.25
data_name=numpy

random_seed=2021
for pred_len in 336 #96 192 336
do
    for decay_scale in 1 #0.5 1 2 5 10
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
          --features S \
          --seq_len $seq_len \
          --pred_len $pred_len \
          --enc_in 21 \
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
          --train_epochs 50\
          --patience 15\
          --itr 1 --batch_size 128 --learning_rate 0.0001 \
          --attn_decay_scale ${decay_scale} \
          "$@" &
          #--attn_decay_type 'gauss' \
          #>logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
    done
done
