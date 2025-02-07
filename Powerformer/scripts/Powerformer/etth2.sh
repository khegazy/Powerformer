if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

model_name=Powerformer
decay_type=powerLaw
decay_scale=0.5
seq_len=336
pred_len=96
itr=0
random_seed=2021

root_path_name=./datasets/
data_path_name=ETT-small/ETTh2.csv
model_id_name=ETTh2
data_name=ETTh2

python3 -u run_longExp.py \
    --attn_decay_type ${decay_type} \
    --attn_decay_scale ${decay_scale} \
    --is_sequential 0 \
    --random_seed ${random_seed} \
    --is_training 1 \
    --root_path ${root_path_name} \
    --data_path ${data_path_name} \
    --model_id ${model_id_name} \
    --model ${model_name} \
    --data ${data_name} \
    --features M \
    --seq_len ${seq_len} \
    --pred_len ${pred_len} \
    --enc_in 7 \
    --e_layers 3 \
    --n_heads 4 \
    --d_model 16 \
    --d_ff 128 \
    --dropout 0.3\
    --fc_dropout 0.3\
    --head_dropout 0\
    --patch_len 16\
    --stride 8\
    --des 'Exp' \
    --train_epochs 100\
    --itr ${itr} \
    --batch_size 128 \
    --learning_rate 0.0001 \
    "$@" >> logs/LongForecasting/${model_name}'_'${model_id_name}'_'${seq_len}'_'${pred_len}.log